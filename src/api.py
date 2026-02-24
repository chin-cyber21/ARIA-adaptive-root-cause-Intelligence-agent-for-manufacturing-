import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ARIA - SAP Manufacturing Defect Intelligence")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(req: QueryRequest):
    from main import run
    result = run(req.question)
    return {
        "intent": result["intent"],
        "answer": result["final_answer"],
        "escalation": result["escalation"],
        "sap_context": result["sap_context"]
    }