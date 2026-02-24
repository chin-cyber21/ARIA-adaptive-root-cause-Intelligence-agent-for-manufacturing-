# Architecture Decisions of ARIA

## 1. Decision of LLM - Groq over Gemini
 
Initially planned to use Gemini it has a token context window of 1M toekns which is great for large to pass data from large manufacturing manuals. 

But into two problems with gemini JSON formatting inconsistencies in responses and rate limiting 
on the free tier which kept breaking the pipeline mid-run.

switched to Groq an open source alternative and runs in cloud so nothing to download or run locally. 
Generous free limits and response format was consistent.

LLM is abstracted through LangChain so swapping to Azure OpenAI in production is just a config change.

## 2. LangGraph over plain LangChain

LangChain works well for individual agent flows like the SAP connector where one agent uses tools in sequence. But when we need orchestration across multiple agents with decision-based routing between them, LangGraph is the right stack. 

ARIA needs to decide at runtime does this query need deep reasoning or can it go straight to synthesis?  
Plain flows can't do this. 
LangGraph holds the shared state and routes between agents based on conditions.

## 3. Hybrid Search over pure Semantic

over this dataset while doing just Semantic search it was missing exact technical column names from the dataset
terms like TWF, HDF, OSF, PWF. These are specific failure indicators and semantic search treats them as unknown tokens, returning vague results.

Added BM25 alongside for exact keyword matching. 
Semantic catches meaning, BM25 catches precision. 

Combined results are deduplicated before passing to agents. Coverage for technical queries improved significantly.

## 4. SAP Connector Agent

Manufacturing plants run on SAP. Wanted to show the real-world integration 
path machines face issues in realtime and plant staff need to cross-check 
maintenance history, open work orders, spare parts stock before making 
decisions. 

SAP connector demonstrates this — even though MVP uses synthetic data, the agent architecture is production-ready. 

In real deployment it would connect to SAP PM module via RFC/BAPI endpoints and decisions can be made based on live plant data. 
Shows the project isn't just a generic RAG demo 
but something that fits actual manufacturing workflows.

## 5. ChromaDB for Vector Storage

Simple to set up runs locally, no external service needed during 
development. Persists to disk so vectorstore doesn't rebuild on every run. 
Production path is Azure AI Search same retrieval interface, just by swapping the backend.

## 6. FastAPI

Lightweight, fast, automatic Swagger docs at /docs and easy to containerize. 
Standard choice for Python ML APIs no reason to use anything heavier for this use case.