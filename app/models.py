"""Pydantic v2 models for chunks, tool results, citations, and API payloads."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A top-level function or class extracted from a Python source file."""

    path: str
    name: str
    kind: Literal["function", "async_function", "class"]
    start_line: int
    end_line: int
    source: str

    @property
    def symbol_id(self) -> str:
        """Return canonical symbol identifier in path::name format."""
        return f"{self.path}::{self.name}"


class ToolResult(BaseModel):
    """Structured search or code-inspection hit returned by tools."""

    path: str
    name: str
    start_line: int
    end_line: int
    score: float | None = None
    snippet: str


class Citation(BaseModel):
    """A file and line-range citation extracted from an agent answer."""

    path: str
    start_line: int
    end_line: int
    verified: bool = False


class TraceStep(BaseModel):
    """A single tool invocation recorded during an agent run."""

    step: int
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    result_summary: str


class Answer(BaseModel):
    """Final verified answer returned by the agent and POST /ask."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    status: Literal["ok", "step_limit", "timeout"] = "ok"
    trace: list[TraceStep] = Field(default_factory=list)


class IndexRequest(BaseModel):
    """Request body for POST /index."""

    path: str | None = None


class IndexStatus(BaseModel):
    """Response body for GET /index/status and POST /index."""

    indexed: bool
    files: int
    chunks: int
    edges: int


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1)
