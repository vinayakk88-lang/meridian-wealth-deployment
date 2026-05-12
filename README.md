# Meridian Wealth — Financial Analyst Agent API

FastAPI service wrapping a LangChain v1 / LangGraph ReAct agent for
Meridian Wealth Partners. The agent has five tools backed by a SQLite
database, a FAISS vector store over policy PDFs, and Tavily web search.

## Quick start

```powershell
python -m venv venv
venv\Scripts\activate              # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env             # then fill in OPENAI_API_KEY and TAVILY_API_KEY

# Provide the SQLite DB under ./data/ (policy PDFs are already committed):
#   data/meridian_wealth.db
#   data/policy_documents/*.pdf   (5 PDFs, included in the repo)

uvicorn app:app --reload
```

First startup embeds the policy PDFs and persists the FAISS index to
`vectorstore/` (~1 min). Subsequent startups load from disk.

Open http://localhost:8000/docs for the OpenAPI UI.

## Endpoints

| Method | Path          | Purpose                                                                |
| ------ | ------------- | ---------------------------------------------------------------------- |
| GET    | `/health`     | Readiness probe — DB, vectorstore, and Tavily-key status.              |
| GET    | `/agent/info` | Model, tool list, RAG config, vectorstore doc count, system prompt.    |
| POST   | `/chat`       | Run the agent. Accepts `message`, optional `conversation_id`, `include_trace`, `max_iterations`. Returns answer + optional tool-call trace. |

## Smoke test

With the server running, in another terminal:

```powershell
python tests\smoke_test.py                    # defaults to http://localhost:8000
python tests\smoke_test.py http://localhost:9000
```

The script hits all three endpoints, asserts 200s and expected payload
shape, prints a one-line summary per check, and exits non-zero on
failure (CI-friendly). If the server isn't running, it prints a hint
instead of a stack trace.

## Project layout

```
app.py                  FastAPI entry point + lifespan
src/
  config.py             env vars, paths, model names
  schemas.py            Pydantic request/response models
  database_queries.py   SQL helpers
  rag_pipeline.py       PDF → split → embed → FAISS
  agent_tools.py        @tool wrappers (factory closes over retriever)
  react_agent.py        System prompt + create_agent factory
data/
  meridian_wealth.db    SQLite DB (gitignored — bring your own)
  policy_documents/     5 policy PDFs (committed)
vectorstore/            persisted FAISS index (gitignored)
tests/
  smoke_test.py         end-to-end endpoint check
```

## Agent tools

| Tool                | Backend                | Purpose                                              |
| ------------------- | ---------------------- | ---------------------------------------------------- |
| `portfolio_lookup`  | SQLite                 | Client holdings, allocation, risk profile.           |
| `market_data_search`| SQLite                 | Stock / sector market data.                          |
| `calculate_metrics` | Pure Python            | Returns, percentages, comparisons.                   |
| `policy_retriever`  | FAISS over policy PDFs | RAG search over investment policies.                 |
| `tavily_search`     | Tavily API             | Live web search for news / market updates.           |

Multi-turn conversations are supported via an `InMemorySaver`
checkpointer keyed by `conversation_id`. State is process-local — swap
in a SQLite/Postgres checkpointer for multi-worker deployments.

## Configuration

All knobs live in `.env` (see `.env.example`):

- `AGENT_MODEL` — OpenAI chat model for the agent (default `gpt-5-mini`).
- `EMBEDDING_MODEL` — embedding model for RAG (default `text-embedding-3-small`).
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVER_K` — RAG tuning.
- `API_HOST`, `API_PORT`, `LOG_LEVEL` — FastAPI runtime.

Required keys: `OPENAI_API_KEY`, `TAVILY_API_KEY`. The lifespan handler
fails fast on startup if either is missing.
