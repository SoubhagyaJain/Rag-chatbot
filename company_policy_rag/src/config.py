"""
Single source of truth for all hyperparameters and paths.

Production rationale:
- Centralized config prevents drift between indexing, agent, and chat app.
- Pydantic Settings allows .env overrides without code changes (12-factor).
- Chunk size 640 tokens balances legal/policy recall (sections stay intact)
  with embedding model context limits (nomic-embed-text ≈ 8192, but smaller
  chunks improve retrieval precision for clause-level questions).
"""

from __future__ import annotations

import os

# Disable Chroma anonymized telemetry before any chromadb import.
# Avoids posthog version mismatch noise: "capture() takes 1 positional argument..."
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_project_root() -> Path:
    """Resolve runtime project root for dev, pip install, and Docker layouts."""
    env_root = os.environ.get("POLICY_RAG_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    dev_root = Path(__file__).resolve().parent.parent
    if (dev_root / "pyproject.toml").is_file() or (dev_root / "data").is_dir():
        return dev_root

    return Path.cwd().resolve()


# Project root: one level above src/ in dev; cwd or POLICY_RAG_ROOT when pip-installed
PROJECT_ROOT = _resolve_project_root()


class Settings(BaseSettings):
    """All tunable parameters for indexing, retrieval, and generation."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    policies_dir: Path = Field(default=PROJECT_ROOT / "data" / "policies")
    legal_dir: Path = Field(default=PROJECT_ROOT / "data" / "legal")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    storage_dir: Path = Field(default=PROJECT_ROOT / "storage")
    pdf_images_dir: Path = Field(default=PROJECT_ROOT / "storage" / "images")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # ── ChromaDB vector store ──────────────────────────────────────────────
    # Chroma replaces SimpleVectorStore: persistent storage, metadata filtering,
    # and incremental indexing without full rebuilds.
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "storage" / "chroma")
    chroma_collection_name: str = Field(
        default="company_policies", alias="CHROMA_COLLECTION_NAME"
    )
    # Distance function for HNSW index: cosine works well with normalized embeddings
    chroma_distance_fn: Literal["cosine", "l2", "ip"] = Field(
        default="cosine", alias="CHROMA_DISTANCE_FN"
    )

    # ── Ollama / Models ────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_LLM_MODEL")
    embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")

    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_request_timeout: float = Field(default=120.0, alias="LLM_REQUEST_TIMEOUT")
    llm_context_window: int = Field(default=8192, alias="LLM_CONTEXT_WINDOW")

    # ── Chunking (tokens) ──────────────────────────────────────────────────
    # 512–768 recommended for policy/legal: keeps clauses together while
    # staying within embedder sweet spot. 640 is our default compromise.
    # Overlap 64 (was 80) reduces near-duplicate chunks that hurt Context Precision.
    chunk_size: int = Field(default=640, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # ── Section detection ──────────────────────────────────────────────────
    # Rich structural metadata is among the highest-ROI improvements for
    # legal/policy RAG — enables better citations, filtering, and reranking.
    enable_section_detection: bool = Field(default=True, alias="ENABLE_SECTION_DETECTION")
    # standard = balanced; strict = formal patterns only; permissive = + ALL CAPS
    section_detection_mode: Literal["standard", "strict", "permissive"] = Field(
        default="standard", alias="SECTION_DETECTION_MODE"
    )
    # Lines to scan at the top of each PDF page for section headers
    section_page_scan_lines: int = Field(default=25)

    # ── Retrieval ──────────────────────────────────────────────────────────
    similarity_top_k: int = Field(default=8, alias="SIMILARITY_TOP_K")
    # Over-retrieve for reranker pool. 25–30 is the sweet spot for policy docs:
    # enough recall for reranker without drowning it in noise.
    retrieval_candidate_k: int = Field(default=30, alias="RETRIEVAL_CANDIDATE_K")

    # ── Reranker (post-retrieval precision) ────────────────────────────────
    # bge-reranker-large > base for Context Precision on dense legal text.
    # Trade-off: ~2× rerank latency vs base. Set RERANKER_MODEL=base for speed.
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large", alias="RERANKER_MODEL"
    )
    # Final context passed to generation (6 balances recall for multi-part policy Q&A)
    reranker_top_n: int = Field(default=6, alias="RERANKER_TOP_N")
    reranker_batch_size: int = Field(default=32, alias="RERANKER_BATCH_SIZE")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    # Drop chunks scoring below this fraction of the top reranker score
    enable_rerank_score_filter: bool = Field(default=True, alias="ENABLE_RERANK_SCORE_FILTER")
    rerank_min_score_ratio: float = Field(default=0.40, alias="RERANK_MIN_SCORE_RATIO")
    rerank_min_keep: int = Field(default=3, alias="RERANK_MIN_KEEP")

    # ── Query rewrite (pre-retrieval) ──────────────────────────────────────
    # Lightweight LLM rewrite improves keyword alignment for policy queries
    enable_query_rewrite: bool = Field(default=True, alias="ENABLE_QUERY_REWRITE")

    # ── Generation / faithfulness grounding ──────────────────────────────────
    # balanced (default): helpful synthesis + partial answers; strict: max faithfulness
    grounding_strictness: Literal["strict", "balanced"] = Field(
        default="balanced", alias="GROUNDING_STRICTNESS"
    )
    response_prompt_version: Literal["v1_standard", "v2_strict", "v2_balanced"] = Field(
        default="v2_balanced", alias="RESPONSE_PROMPT_VERSION"
    )
    # Legacy override — true forces strict mode regardless of GROUNDING_STRICTNESS
    strict_grounding: bool = Field(default=False, alias="STRICT_GROUNDING")
    enable_faithfulness_check: bool = Field(default=True, alias="ENABLE_FAITHFULNESS_CHECK")
    # strict guard: reject unless fully supported; balanced: reject only clear hallucinations
    faithfulness_guard_mode: Literal["strict", "balanced", "off"] = Field(
        default="balanced", alias="FAITHFULNESS_GUARD_MODE"
    )

    # ── Agent ──────────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(default=8)
    agent_verbose: bool = Field(default=True)

    # ── Conversation memory (short-term, per session) ──────────────────────
    # Enables natural follow-ups ("What about part-time?") without external stores.
    enable_conversation_memory: bool = Field(default=True, alias="ENABLE_CONVERSATION_MEMORY")
    memory_window_size: int = Field(default=5, alias="MEMORY_WINDOW_SIZE")  # turns (user+assistant pairs)
    memory_token_limit: int = Field(default=3000, alias="MEMORY_TOKEN_LIMIT")

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # ── Chat UI ────────────────────────────────────────────────────────────
    chainlit_port: int = Field(default=8000, alias="CHAINLIT_PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    # ── Citation display (chat UI) ─────────────────────────────────────────
    # Citations are critical for trust in policy/legal RAG — keep configurable
    # so UX can evolve (inline vs sidebar, excerpts on/off) without code changes.
    show_citations: bool = Field(default=True, alias="SHOW_CITATIONS")
    citation_max_sources: int = Field(default=6, alias="CITATION_MAX_SOURCES")
    citation_show_excerpts: bool = Field(default=True, alias="CITATION_SHOW_EXCERPTS")
    citation_show_relevance_score: bool = Field(default=False, alias="CITATION_SHOW_SCORE")
    citation_dedupe: bool = Field(default=True, alias="CITATION_DEDUPE")
    # section_first: "II. GENERAL > 5.2 Vacation (p.14)"
    # document_first: "Employee Handbook.pdf — Section 5.2 (Page 14)"
    citation_format: Literal["section_first", "document_first"] = Field(
        default="section_first", alias="CITATION_FORMAT"
    )
    # Minimum reranker score (fraction of top chunk) to show a source when the
    # answer has no explicit [Source N] tags. Higher = fewer, more precise citations.
    citation_min_relevance_ratio: float = Field(
        default=0.55, alias="CITATION_MIN_RELEVANCE_RATIO"
    )
    # Log retrieved / filtered / displayed chunks for citation debugging.
    enable_citation_pipeline_logging: bool = Field(
        default=True, alias="ENABLE_CITATION_PIPELINE_LOGGING"
    )

    # ── Evaluation ───────────────────────────────────────────────────────
    # Golden-set eval is the quality gate before chunking / retrieval changes.
    eval_dataset_path: Path = Field(default=PROJECT_ROOT / "data" / "eval" / "golden_dataset.json")
    eval_results_path: Path = Field(default=PROJECT_ROOT / "logs" / "evaluation_results.json")
    eval_llm_model: str = Field(default="qwen2.5:7b", alias="EVAL_LLM_MODEL")
    eval_max_samples: int = Field(default=0, alias="EVAL_MAX_SAMPLES")  # 0 = all cases
    eval_use_llm_judge: bool = Field(default=True, alias="EVAL_USE_LLM_JUDGE")

    # ── PDF image extraction (citation visuals) ────────────────────────────
    enable_pdf_images: bool = Field(default=True, alias="ENABLE_PDF_IMAGES")
    pdf_image_min_px: int = Field(default=80, alias="PDF_IMAGE_MIN_PX")
    pdf_page_thumb_dpi: int = Field(default=120, alias="PDF_PAGE_THUMB_DPI")
    citation_max_page_images: int = Field(default=4, alias="CITATION_MAX_PAGE_IMAGES")

    # ── Comprehensive list retrieval ───────────────────────────────────────
    enable_comprehensive_retrieval: bool = Field(
        default=True, alias="ENABLE_COMPREHENSIVE_RETRIEVAL"
    )
    comprehensive_reranker_top_n: int = Field(
        default=10, alias="COMPREHENSIVE_RERANKER_TOP_N"
    )
    comprehensive_max_subqueries: int = Field(default=8, alias="COMPREHENSIVE_MAX_SUBQUERIES")

    # ── Document taxonomy ──────────────────────────────────────────────────
    # Maps top-level data subfolders → metadata document_type values
    folder_document_types: dict[str, str] = Field(
        default_factory=lambda: {
            "policies": "company_policy",
            "legal": "legal_document",
            "raw": "raw_backup",
        }
    )

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not exist."""
        for path in (
            self.data_dir,
            self.policies_dir,
            self.legal_dir,
            self.raw_dir,
            self.storage_dir,
            self.pdf_images_dir,
            self.chroma_persist_dir,
            self.logs_dir,
            self.eval_dataset_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()