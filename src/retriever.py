import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4
TOP_K = 5


def hybrid_search(query: str, vs: Chroma, chunks: List[Document]) -> List[Document]:
    """
    Manual hybrid search - combines semantic + BM25 scores.
    Semantic alone misses exact terms like TWF, HDF.
    BM25 alone misses meaning. Together they catch everything.
    """
    # semantic search
    semantic_results = vs.similarity_search(query, k=TOP_K)
    
    # keyword search
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    bm25_results = bm25.invoke(query)

    # combine - deduplicate by content
    seen = set()
    combined = []
    
    for doc in semantic_results + bm25_results:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    return combined[:TOP_K]


def get_retriever_components():
    """Returns vs and chunks needed for hybrid search."""
    from src.vectorstore import load_vectorstore
    from src.ingestion import load_and_chunk_all
    
    vs = load_vectorstore()
    chunks = load_and_chunk_all()
    return vs, chunks


if __name__ == "__main__":
    vs, chunks = get_retriever_components()
    
    query = "bearing failure high torque"
    results = hybrid_search(query, vs, chunks)
    
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] {doc.page_content[:150]}")