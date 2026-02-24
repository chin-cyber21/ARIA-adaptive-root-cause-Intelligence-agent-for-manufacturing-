import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import aria
from src.cache import get_cached, set_cache


def run(query: str) -> dict:
    # checking cache memory first
    cached = get_cached(query)
    if cached:
        print("cache hit")
        return cached

    result = aria.invoke({
        "query": query,
        "intent": "",
        "intent_confidence": 0.0,
        "retrieved_docs": [],
        "retrieval_confidence": 0.0,
        "sap_context": {},
        "reasoning": "",
        "final_answer": {},
        "escalation": {},
        "iterations": 0,
        "error": None
    })

    set_cache(query, result)
    return result


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Why is M001 showing bearing failure?"
    result = run(query)
    print("\n ARIA ")
    print(f"intent:    {result['intent']}")
    print(f"answer:    {result['final_answer']}")
    print(f"escalation: {result['escalation']}")