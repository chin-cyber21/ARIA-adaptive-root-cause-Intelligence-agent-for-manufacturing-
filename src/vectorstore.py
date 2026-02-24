import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aria_manufacturing")


def get_embeddings():
    # free, open source runs locally on my pc no costing 
    # MiniLM is fast and good enough for technical docs
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    # Embed all chunks and store in ChromaDB.

    print(f"Embedding {len(chunks)} chunks, this takes a few minutes...")
    embeddings = get_embeddings()

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"Done. Saved to {CHROMA_DB_PATH}")
    return vs


def load_vectorstore() -> Chroma:
    """Load existing vectorstore from disk - fast, no re-embedding."""
    print("Loading vectorstore from disk...")
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def vectorstore_exists() -> bool:
    return os.path.exists(CHROMA_DB_PATH)


if __name__ == "__main__":
    from ingestion import load_and_chunk_all

    if vectorstore_exists():
        print("Vectorstore already exists, loading...")
        vs = load_vectorstore()
    else:
        chunks = load_and_chunk_all()
        vs = build_vectorstore(chunks)

    # test output / checkmark 
    results = vs.similarity_search("bearing failure high torque", k=3)
    print(f"\nTest search: {len(results)} results")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] {doc.page_content[:150]}")