import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# final structured output - what the user actually sees
class ARIAResponse(BaseModel):
    root_cause: str
    confidence: float
    immediate_action: str
    source_reference: str
    escalate: bool
    summary: str


def synthesize_response(state: dict) -> dict:
    # pull everything from state
    docs_text = "\n\n".join(state.get("retrieved_docs", []))
    reasoning = state.get("reasoning", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are ARIA, a manufacturing defect intelligence assistant.
Generate a structured response based on the analysis.
Return valid JSON only:
{{
    "root_cause": "primary cause of the issue",
    "confidence": 0.0-1.0,
    "immediate_action": "what to do right now",
    "source_reference": "which data records support this",
    "escalate": true/false,
    "summary": "2-3 line summary for the technician"
}}
"""),
        ("human", """Query: {query}
        
Data: {docs}

Analysis: {reasoning}

Generate structured response.""")
    ])

    result = (prompt | llm).invoke({
        "query": state["query"],
        "docs": docs_text,
        "reasoning": reasoning
    })

    try:
        text = result.content.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(text)
    except Exception:
        # fallback response if parsing fails
        parsed = {
            "root_cause": "Unable to determine",
            "confidence": 0.3,
            "immediate_action": "Manual inspection recommended",
            "source_reference": "N/A",
            "escalate": True,
            "summary": "Could not generate structured response, please consult maintenance team"
        }

    print(f"  → synthesis done (confidence: {parsed.get('confidence', 0)})")

    return {
        **state,
        "final_answer": parsed
    }