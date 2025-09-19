from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


def build_chain(persist_directory: Path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context to answer.\n"
                "If the answer isn't in the context, say you don't know concisely.\n"
                "Keep answers brief and cite sources as filenames when relevant.",
            ),
            ("human", "Question: {input}\n\nContext:\n{context}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain


def format_sources(documents: List[Document]) -> List[str]:
    seen = set()
    ordered_sources: List[str] = []
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            ordered_sources.append(src)
    return ordered_sources


load_dotenv()
app = FastAPI(title="RAG App (LangChain + Chroma + Gemini)")

# CORS for frontend (Next.js dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    persist_directory = Path(__file__).parent / "chroma"
    if not persist_directory.exists():
        raise RuntimeError(
            f"Chroma directory not found at {persist_directory}. Run ingest.py first."
        )
    # Attach the chain to the app state for reuse
    app.state.rag_chain = build_chain(persist_directory)
    # Keep vector store and embeddings accessible for real-time upserts
    app.state.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    app.state.vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=app.state.embeddings,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Minimal HTML UI
    html = """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>RAG App</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; max-width: 800px; }
      h1 { font-size: 20px; margin-bottom: 12px; }
      input[type=text] { width: 100%; padding: 10px; font-size: 14px; }
      button { margin-top: 12px; padding: 10px 16px; font-size: 14px; }
      .answer { margin-top: 20px; padding: 12px; background: #f7f7f7; }
      .sources { margin-top: 8px; font-size: 12px; color: #444; }
      .error { color: #b00020; margin-top: 12px; }
    </style>
  </head>
  <body>
    <h1>RAG App (LangChain + Chroma + Gemini)</h1>
    <input id=\"q\" type=\"text\" placeholder=\"Ask a question...\" />
    <button onclick=\"ask()\">Ask</button>
    <div id=\"err\" class=\"error\"></div>
    <div id=\"ans\" class=\"answer\"></div>
    <div id=\"src\" class=\"sources\"></div>

    <script>
      async function ask() {
        const err = document.getElementById('err');
        const ans = document.getElementById('ans');
        const src = document.getElementById('src');
        err.textContent = '';
        ans.textContent = '';
        src.textContent = '';
        const q = document.getElementById('q').value.trim();
        if (!q) { err.textContent = 'Please enter a question.'; return; }
        try {
          const res = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: q })
          });
          if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || ('HTTP ' + res.status));
          }
          const data = await res.json();
          ans.textContent = data.answer || '';
          if (data.sources && data.sources.length) {
            src.textContent = 'Sources:\n' + data.sources.join('\n');
          }
        } catch (e) {
          err.textContent = 'Error: ' + (e?.message || e);
        }
      }
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    try:
        result = app.state.rag_chain.invoke({"input": req.query})
        answer: str = result.get("answer", "")
        context_docs: List[Document] = result.get("context", [])
        sources = format_sources(context_docs)
        return AskResponse(answer=answer.strip(), sources=sources)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    documents: list[Document] = []
    for f in files:
        try:
            content_type = f.content_type or ""
            data = await f.read()
            text = ""
            if content_type == "application/pdf" or (f.filename or "").lower().endswith(".pdf"):
                try:
                    reader = PdfReader(io.BytesIO(data))  # type: ignore[name-defined]
                except NameError:
                    import io  # lazy import
                    reader = PdfReader(io.BytesIO(data))
                pages = []
                for page in reader.pages:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
                text = "\n\n".join(pages).strip()
            else:
                # Treat as UTF-8 text
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""

            if not text:
                continue

            documents.append(
                Document(page_content=text, metadata={"source": f.filename or "upload"})
            )
        finally:
            await f.close()

    if not documents:
        raise HTTPException(status_code=400, detail="No readable text from uploads")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    # Upsert into vector store
    app.state.vector_store.add_documents(chunks)

    return JSONResponse({
        "files": [d.metadata.get("source", "upload") for d in documents],
        "chunks_added": len(chunks),
    })


