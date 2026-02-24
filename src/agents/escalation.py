import os
from dotenv import load_dotenv

load_dotenv()

# these thresholds came from looking at the dataset patterns
LOW_STOCK_THRESHOLD = 3
HIGH_WORKORDER_THRESHOLD = 3


def escalation_agent(state: dict) -> dict:
    answer = state.get("final_answer", {})
    sap = state.get("sap_context", {})

    should_escalate = False
    reasons = []

    # low confidence = uncertain diagnosis = escalate
    if answer.get("confidence", 1) < 0.8:
        should_escalate = True
        reasons.append("low diagnosis confidence")

    # synthesis itself flagged it
    if answer.get("escalate", False):
        should_escalate = True
        reasons.append("critical failure pattern detected")

    # check SAP stock - can't fix without parts
    if sap.get("found") and "data" in sap:
        sap_str = sap["data"]
        # rough parse - good enough for MVP
        if "Bearing stock: 1" in sap_str or "Bearing stock: 2" in sap_str:
            should_escalate = True
            reasons.append("low bearing stock")

        if "Open work orders: 3" in sap_str or "Open work orders: 4" in sap_str:
            should_escalate = True
            reasons.append("high open work orders")

    report = {
        "escalate": should_escalate,
        "reasons": reasons,
        "priority": "HIGH" if should_escalate else "NORMAL",
        "action": "contact Level 2 maintenance immediately" if should_escalate else "schedule routine check"
    }

    print(f" priority: {report['priority']}")

    return {**state, "escalation": report}