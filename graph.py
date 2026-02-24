import os
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# importing all agents
from src.agents.classifier import classify_intent
from src.agents.retrieval import retrieve_documents
from src.agents.sap_agent import sap_connector
from src.agents.reasoning import reason_over_docs
from src.agents.synthesis import synthesize_response
from src.agents.escalation import escalation_agent


# shared state - every agent reads and writes to this
class ARIAState(TypedDict):
    query: str
    intent: str
    intent_confidence: float
    retrieved_docs: List[str]
    retrieval_confidence: float
    sap_context: dict
    reasoning: str
    final_answer: dict
    escalation: dict
    iterations: int
    error: Optional[str]


def route_after_classifier(state: ARIAState) -> str:
    intent = state.get("intent", "simple_lookup")
    # complex queries need reasoning, simple ones go straight to synthesis
    if intent in ["root_cause", "historical_pattern"]:
        return "reasoning"
    return "synthesis"


def should_retry(state: ARIAState) -> str:
    # retry retrieval if confidence too low, max 3 times
    if state.get("retrieval_confidence", 1) < 0.4 and state.get("iterations", 0) < 3:
        return "retry"
    return "continue"


def build_graph():
    graph = StateGraph(ARIAState)

    # register all nodes
    graph.add_node("classifier", classify_intent)
    graph.add_node("retrieval", retrieve_documents)
    graph.add_node("sap", sap_connector)
    graph.add_node("reasoning", reason_over_docs)
    graph.add_node("synthesis", synthesize_response)
    graph.add_node("escalation", escalation_agent)

    # entry point
    graph.set_entry_point("classifier")

    # classifier to retrieval always
    graph.add_edge("classifier", "retrieval")

    # retrieval to retry or continue based on confidence
    graph.add_conditional_edges(
        "retrieval",
        should_retry,
        {
            "retry": "retrieval",   # loop back
            "continue": "sap"       # move forward
        }
    )

    # sap to classifier decides next (reasoning or synthesis)
    graph.add_conditional_edges(
        "sap",
        route_after_classifier,
        {
            "reasoning": "reasoning",
            "synthesis": "synthesis"
        }
    )

    # reasoning to synthesis always
    graph.add_edge("reasoning", "synthesis")

    # synthesis to escalation to end
    graph.add_edge("synthesis", "escalation")
    graph.add_edge("escalation", END)

    return graph.compile()


# compiled graph - imported by api.py and main.py
aria = build_graph()


if __name__ == "__main__":
    print("Testing ARIA pipeline...\n")

    result = aria.invoke({
        "query": "Why is machine M001 showing bearing failure with high torque?",
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

    print("\n=== ARIA RESPONSE ===")
    print(f"Intent: {result['intent']}")
    print(f"SAP Context: {result['sap_context']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Escalation: {result['escalation']}")