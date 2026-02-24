import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# keeping temperature 0 here - we want consistent routing
# flaky intent detection breaks the whole pipeline
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def classify_intent(state: dict) -> dict:
# function Classifies query intent to route to the correct agent.

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a manufacturing query classifier.
Classify the query into exactly one of these:

- root_cause: why did something fail
- repair_procedure: how to fix something
- historical_pattern: has this happened before
- simple_lookup: specific fact or value lookup

Return valid JSON only:
{{"intent": "one of above", "confidence": 0.0-1.0, "reasoning": "one line"}}
"""),
        ("human", "{query}")
    ])

    result = (prompt | llm).invoke({"query": state["query"]})

    try:
        # gemini output text cleaning 
        text = result.content.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(text)
    except Exception:
        # adding fallback for syntax mixmatch
        parsed = {
            "intent": "simple_lookup",
            "confidence": 0.5,
            "reasoning": "parse failed"
        }

    print(f" intent: {parsed['intent']} ({parsed['confidence']})")

    return {
        **state,
        "intent": parsed["intent"],
        "intent_confidence": parsed["confidence"]
    }