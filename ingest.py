import argparse
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_text_files_from_directory(data_directory: Path) -> List[Document]:
    """Read all .txt files from a directory and return LangChain Documents.

    Each document includes minimal metadata with the file path.
    """
    documents: List[Document] = []
    for text_file_path in sorted(data_directory.glob("*.txt")):
        file_content: str = text_file_path.read_text(encoding="utf-8")
        documents.append(
            Document(page_content=file_content, metadata={"source": str(text_file_path)})
        )
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents into overlapping segments suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], persist_directory: Path) -> None:
    """Create a Chroma vector store from chunks and persist it to disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    # Ensure on-disk persistence
    vector_store.persist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data into Chroma vector store")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Directory containing .txt files to ingest",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(Path(__file__).parent / "chroma"),
        help="Directory to persist Chroma DB",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector store before ingesting",
    )
    args = parser.parse_args()

    load_dotenv()  # Load OPENAI_API_KEY if present

    data_directory = Path(args.data_dir)
    persist_directory = Path(args.persist_dir)

    if args.reset and persist_directory.exists():
        shutil.rmtree(persist_directory)

    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")

    raw_documents = read_text_files_from_directory(data_directory)
    if not raw_documents:
        raise RuntimeError(
            f"No .txt files found in {data_directory}. Add files and try again."
        )

    chunks = chunk_documents(raw_documents)
    build_vector_store(chunks, persist_directory)
    print(f"Ingestion complete. Persisted to: {persist_directory}")


if __name__ == "__main__":
    main()


