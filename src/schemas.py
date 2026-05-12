"""schemas.py — Pydantic request/response models for /chat and /health.

These are the only Pydantic models in the project; the rest of the code
uses plain dicts and dataclasses to keep object-orientation minimal.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- /chat ---------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's natural-language query for the agent.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Thread ID for multi-turn. If absent, server generates a UUID4.",
    )
    include_trace: bool = Field(
        default=False,
        description="If true, response includes the full tool-call trace.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional cap on agent ReAct loop steps (safety guard).",
    )


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    output: str


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    tool_calls: list[ToolCall] | None = None
    iteration_count: int
    model: str


# --- Errors --------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    conversation_id: str | None = None


# --- /health -------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db_connected: bool
    vectorstore_loaded: bool
    tavily_configured: bool


# --- /agent/info ---------------------------------------------------------

class ToolInfo(BaseModel):
    name: str
    description: str


class AgentInfo(BaseModel):
    model: str
    embedding_model: str
    tools: list[ToolInfo]
    rag: dict[str, int]
    vectorstore_docs: int | None = None
    system_prompt: str
