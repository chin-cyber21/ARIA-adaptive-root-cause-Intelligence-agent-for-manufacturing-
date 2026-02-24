import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv


load_dotenv()

TOP_K = 5


def retrieve_documents(state: dict) -> dict:
    """
    Hybrid search - semantic + BM25 combined.
    Gets called after classifier sets the intent.
    intent is acting as the identifier to identify what doc to call
    """
    from src.vectorstore import load_vectorstore
    from src.ingestion import load_and_chunk_all

    query = state["query"]
    vs = load_vectorstore()
    chunks = load_and_chunk_all()

    # semantic search
    semantic_results = vs.similarity_search(query, k=TOP_K)

    # keyword search (for better searching output)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    bm25_results = bm25.invoke(query)

    # merge and deduplicate
    seen = set()
    combined = []
    for doc in semantic_results + bm25_results:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    final = combined[:TOP_K]
    
    # confidence based on how many results we got
    confidence = len(final) / TOP_K

    print(f"  → retrieved {len(final)} docs (confidence: {confidence})")

    return {
        **state,
        "retrieved_docs": [doc.page_content for doc in final],
        "retrieval_confidence": confidence,
        "iterations": state.get("iterations", 0) + 1
    }