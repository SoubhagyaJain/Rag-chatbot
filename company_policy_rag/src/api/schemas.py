"""Pydantic request/response models for the Aether API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CorpusScope = Literal["all", "policy", "guidebook"]
ChatMode = Literal["direct", "agent"]
GroundingMode = Literal["balanced", "strict"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    corpus_scope: CorpusScope = "all"
    chat_mode: ChatMode = "direct"
    llm_model: str | None = None
    grounding_mode: GroundingMode | None = None


class CitationOut(BaseModel):
    source_file: str | None = None
    section_path: str | None = None
    page_number: int | None = None
    excerpt: str | None = None
    relevance_score: float | None = None
    selection_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    timing: dict[str, Any] | None = None
    low_confidence: bool = False
    grounding_mode: str = "balanced"
    thinking: str | None = None
    retrieval_trace: dict[str, Any] | None = None
    message_id: str | None = None


class ModelInfo(BaseModel):
    id: str
    label: str
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    badges: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    rating: Literal[1, -1]
    question: str
    answer: str
    model: str = ""
    corpus_scope: CorpusScope = "all"
    message_id: str | None = None
    comment: str | None = None


class ModelsResponse(BaseModel):
    connected: bool
    error: str | None = None
    active_model: str
    models: list[ModelInfo]


class SetModelRequest(BaseModel):
    model: str


class HealthResponse(BaseModel):
    index_ready: bool
    chunk_count: int = 0
    collection: str | None = None
    last_updated: str | None = None
    ollama_connected: bool
    ollama_error: str | None = None
    llm_model: str
    embed_model: str
    retrieval: dict[str, Any] = Field(default_factory=dict)


class EvalRunRequest(BaseModel):
    max_samples: int | None = Field(default=5, ge=1, le=100)


class EvalJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    result: dict[str, Any] | None = None
    error: str | None = None