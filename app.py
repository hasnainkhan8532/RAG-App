import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def format_sources(documents: List[Document]) -> str:
    unique_sources = []
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        if src not in unique_sources:
            unique_sources.append(src)
    return "\n".join(unique_sources)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple CLI RAG App (LangChain + Chroma)")
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(Path(__file__).parent / "chroma"),
        help="Directory of persisted Chroma DB",
    )
    parser.add_argument("--query", type=str, default=None, help="Query to ask the RAG app")
    args = parser.parse_args()

    load_dotenv()

    persist_directory = Path(args.persist_dir)
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma directory not found at {persist_directory}. Run ingest.py first."
        )

    chain = build_chain(persist_directory)

    query = args.query
    if not query:
        try:
            query = input("Enter your question: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            return

    result = chain.invoke({"input": query})
    answer: str = result.get("answer", "")
    context_docs: List[Document] = result.get("context", [])

    print("\nAnswer:\n" + answer.strip())
    if context_docs:
        print("\nSources:")
        print(format_sources(context_docs))


if __name__ == "__main__":
    main()


