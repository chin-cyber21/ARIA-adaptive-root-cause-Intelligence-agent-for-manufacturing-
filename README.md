# ARIA : adaptive Root cause intelligent agent 

Aria is a intelligent multi agent system that is designed to help manufacturing engineers and manufacturing plant staff to diagnose machine failures using natural language minimizing the risk of any unhappening by pulling the relevant history from machine records and historic maintenance data.

Ask aria why a machine is failing or an uncertain activity is happening it pulls relevant data from machine records, SAP maintenance data and returns a structured answer with escalation priority. 

I have built this project for manufacturing usecase as most of the ai tools in this segment are inefficeint or too expesive. 

## Architecture 

Aria cosnsit of 6 different agents/ functions orchestrated via LangGraph, each agent has its role and intent as follows: 

a. classifier : Figures out the user query intent 
b. Retrieval : makes a hybrid search ( BM25 keyword + semantic ) over machine records 
c. Sap Connector: custom tools based agent created to fetch real data (workcase demonstrated with syntetic data in mvp) and perform operation/ tool calling based on intent. 
d. Reasoning: deep analysis for complex root cause queries via user
e. Escalationn: Decides the priority of situation based on confidence and stock levels 
f. Synthesis: Generates the final structured output answer with relevant confidence score

state flows through all agents via shared TypeDict (LangGraph)

## stack 

Agentic framework : Langchain, LangGraph
vector storage: ChromaDB , Sentence-transformers ( local embedding)
LLM Reasoning: Groq llama3-70b (can be swappable to Azure OpenAI in production)
Dataset: AI4I 2020 Predictive Maintenance ( Kaggle), custom build sap_maintenance.csv dataset 
framework: FastAPi
Evaluation: Ragas, manual testing 

## setup 

git clone 
cd ARIA 
python -m venv venv 
venv\Scripts\activate 
pip install -r requirements.txt
cp .env.example as .env (get api keys)

Build Vectorstore (initial setup only):
~ python src/vectorstore.py 

Run pipeline
``` bash 
python src/vectorstore.py
```
Run Api
``` bash
uvicorn src.api:app --reload --port 8000
```
swagger : `http://localhost:8000/docs`

## Evaluation 
```bash 
python evaluation/test_suite.py
pyhton evaluation/ragas_eval.py
```

## Production Path 
Local : ChromaDB, Groq(used)/Gemini , FastAPI 
production: Azure Ai Search, Azure OpenAI, Azure Container Apps

Config change only — no code rewrite needed for production migration.
