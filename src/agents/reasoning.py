import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)


def reason_over_docs(state: dict) -> dict:
    # only runs for root_cause and historical_pattern queries
    # takes retrieved docs + SAP context and thinks through them
    docs_text = "\n\n".join(state.get("retrieved_docs", []))
    sap_text = str(state.get("sap_context", {}).get("data", "no SAP data"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior manufacturing engineer. "
                   "Analyze the machine data and identify failure patterns. "
                   "Think step by step and be specific."),
        ("human", "Query: {query}\n\nMachine Data:\n{docs}\n\nSAP Context:\n{sap}")
    ])

    result = (prompt | llm).invoke({
        "query": state["query"],
        "docs": docs_text,
        "sap": sap_text
    })

    print(" reasoning complete")

    return {**state, "reasoning": result.content}