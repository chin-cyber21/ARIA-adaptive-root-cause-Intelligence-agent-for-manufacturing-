import os
import pandas as pd
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# TODO: replace this with actual SAP RFC calls ( in production/ reallife)
SAP_DATA_PATH = "./data/raw/sap_maintenance.csv"


@tool
def query_sap_maintenance(machine_id: str) -> str:
    """Get SAP maintenance history for a specific machine."""
    try:
        df = pd.read_csv(SAP_DATA_PATH)
        df.columns = df.columns.str.strip()
        
        # find the machine
        machine_data = df[df['machine_id'] == machine_id]
        
        if machine_data.empty:
            return f"no data found for machine {machine_id}"
            
        row = machine_data.iloc[0]
        return (f"Machine {machine_id} | "
                f"Last maintenance: {row['last_maintenance']} | "
                f"Open work orders: {row['open_work_orders']} | "
                f"Bearing stock: {row['bearing_stock']} | "
                f"Hydraulic stock: {row['hydraulic_stock']} | "
                f"Status: {row['status']}")
    except Exception as e:
        return f"error querying SAP: {str(e)}"


@tool
def get_all_critical_machines() -> str:
    """Get all machines currently in critical status."""
    try:
        df = pd.read_csv(SAP_DATA_PATH)
        df.columns = df.columns.str.strip()
        
        critical = df[df['status'] == 'critical']
        
        if critical.empty:
            return "no critical machines"
            
        machines = []
        for _, row in critical.iterrows():
            machines.append(f"{row['machine_id']} ({row['open_work_orders']} open orders)")
            
        return "critical machines: " + ", ".join(machines)
    except Exception as e:
        return f"error: {str(e)}"


# setup LLM with tools
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
llm_with_tools = llm.bind_tools([query_sap_maintenance, get_all_critical_machines])


def sap_connector(state: dict) -> dict:
    """Call SAP tools based on the query."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a SAP connector. Use tools to fetch relevant maintenance data."),
        ("human", "{query}")
    ])

    result = (prompt | llm_with_tools).invoke({"query": state["query"]})

    sap_context = {"found": False}
    
    # check if LLM decided to use any tools
    if hasattr(result, "tool_calls") and result.tool_calls:
        outputs = []
        for tool_call in result.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            
            print(f"  → calling SAP tool: {name}")
            
            if name == "query_sap_maintenance":
                out = query_sap_maintenance.invoke(args)
            elif name == "get_all_critical_machines":
                out = get_all_critical_machines.invoke(args)
            else:
                out = f"unknown tool: {name}"
                
            outputs.append(f"{name}: {out}")
            
        sap_context = {"found": True, "data": "\n".join(outputs)}
    else:
        print("  → no SAP tools called")

    return {**state, "sap_context": sap_context}


if __name__ == "__main__":
    # quick test
    print("Testing M001:", query_sap_maintenance.invoke({"machine_id": "M001"}))
    print("Critical:", get_all_critical_machines.invoke({}))