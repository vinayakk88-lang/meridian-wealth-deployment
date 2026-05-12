"""app.py — FastAPI entry point.

Lifespan handler builds (or loads) the FAISS index and instantiates the
agent once per process. The /chat endpoint invokes the agent asynchronously
(via ainvoke); /health is a readiness probe.

Run:
    uvicorn app:app --reload                 # dev
    python app.py                            # uses host/port from .env
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.agent_tools import build_tools
from src.config import (
    AGENT_MODEL,
    API_HOST,
    API_PORT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LOG_LEVEL,
    RETRIEVER_K,
    assert_required_keys,
)
from src.database_queries import check_connection
from src.rag_pipeline import get_or_build_vectorstore, get_retriever
from src.react_agent import SYSTEM_PROMPT, build_agent
from src.schemas import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ToolCall,
    ToolInfo,
)


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("app")


# --- Lifespan ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build agent + vectorstore on startup; nothing to tear down."""
    logger.info("Verifying required API keys...")
    assert_required_keys()

    logger.info("Loading / building FAISS index...")
    vectorstore = get_or_build_vectorstore()
    retriever = get_retriever(vectorstore)

    logger.info("Wiring agent tools...")
    tools = build_tools(retriever)

    logger.info("Building ReAct agent (model=%s)...", AGENT_MODEL)
    agent = build_agent(tools)

    app.state.agent = agent
    app.state.vectorstore = vectorstore
    app.state.tools = tools
    logger.info("Startup complete ✅")

    yield


# --- App + middleware ----------------------------------------------------

app = FastAPI(
    title="Meridian Wealth — Financial Analyst Agent API",
    description="LangGraph ReAct agent over SQL + FAISS RAG + Tavily web search.",
    version="0.1.0",
    lifespan=lifespan,
)

# Tighten allow_origins to your frontend domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers -------------------------------------------------------------

def _extract_tool_calls(messages: list[Any]) -> list[ToolCall]:
    """Pair AIMessage tool_calls with their corresponding ToolMessage outputs.

    LangGraph emits an AIMessage with a tool_calls list, then a ToolMessage
    per call (linked via tool_call_id). We zip them so the trace shows each
    tool's name + arguments + output together.
    """
    pending: dict[str, dict[str, Any]] = {}     # tool_call_id → {name, arguments}
    completed: list[ToolCall] = []

    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                pending[tc["id"]] = {"name": tc["name"], "arguments": tc.get("args", {})}
        elif msg_type == "ToolMessage":
            entry = pending.pop(getattr(msg, "tool_call_id", ""), None)
            if entry is None:
                # Couldn't pair — still record the output
                completed.append(ToolCall(
                    name=getattr(msg, "name", "unknown"),
                    arguments={},
                    output=str(msg.content)[:2000],
                ))
                continue
            completed.append(ToolCall(
                name=entry["name"],
                arguments=entry["arguments"],
                output=str(msg.content)[:2000],
            ))
    return completed


# --- Routes --------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    vs_loaded = getattr(request.app.state, "vectorstore", None) is not None
    db_ok = check_connection()
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    overall = "ok" if (vs_loaded and db_ok and tavily_ok) else "degraded"
    return HealthResponse(
        status=overall,
        db_connected=db_ok,
        vectorstore_loaded=vs_loaded,
        tavily_configured=tavily_ok,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    conversation_id = req.conversation_id or str(uuid.uuid4())

    invoke_input = {"messages": [{"role": "user", "content": req.message}]}
    config: dict[str, Any] = {"configurable": {"thread_id": conversation_id}}
    if req.max_iterations is not None:
        # LangGraph counts super-steps; ReAct uses ~2 per iteration.
        config["recursion_limit"] = req.max_iterations * 2 + 1

    try:
        result = await agent.ainvoke(invoke_input, config=config)
    except Exception as exc:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    messages = result["messages"]
    final = messages[-1]
    answer = final.content if hasattr(final, "content") else str(final)

    tool_calls = _extract_tool_calls(messages) if req.include_trace else None
    iteration_count = sum(1 for m in messages if type(m).__name__ == "ToolMessage")

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        tool_calls=tool_calls,
        iteration_count=iteration_count,
        model=AGENT_MODEL,
    )


@app.get("/agent/info", response_model=AgentInfo)
async def agent_info(request: Request) -> AgentInfo:
    tools = getattr(request.app.state, "tools", None) or []
    vectorstore = getattr(request.app.state, "vectorstore", None)
    doc_count = vectorstore.index.ntotal if vectorstore is not None else None

    return AgentInfo(
        model=AGENT_MODEL,
        embedding_model=EMBEDDING_MODEL,
        tools=[
            ToolInfo(name=t.name, description=(t.description or "").strip())
            for t in tools
        ],
        rag={
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "retriever_k": RETRIEVER_K,
        },
        vectorstore_docs=doc_count,
        system_prompt=SYSTEM_PROMPT,
    )


# --- Direct-run entrypoint ----------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False)
