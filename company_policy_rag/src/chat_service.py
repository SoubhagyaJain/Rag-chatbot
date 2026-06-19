"""Framework-agnostic chat orchestration for Streamlit and FastAPI."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from src.agent import AgentTurnResult, chat_with_memory, configure_llm, create_agent
from src.citations import (
    begin_citation_turn,
    get_generation_nodes_this_turn,
    select_citations_for_answer,
)
from src.config import settings
from src.indexing import (
    configure_llama_index,
    get_collection_stats,
    get_or_create_index,
    probe_chroma_index,
)
from src.language import append_language_hint
from src.memory import create_session_memory
from src.ollama_client import filter_chat_models, format_model_label, probe_ollama_tags
from src.prompts import LOW_CONFIDENCE_MESSAGE, resolve_grounding_mode
from src.retrieval_scope import corpus_retrieval_filters, resolve_query_filters
from src.retriever import build_query_engine, build_retriever, get_retrieval_config_summary
from src.timing import begin_query_timing, clear_timing, get_current_timing, record_stage
from src.utils import logger

CorpusScope = Literal["all", "policy", "guidebook"]
ChatMode = Literal["direct", "agent"]
GroundingMode = Literal["balanced", "strict"]

_EMBED_MARKERS = ("embed", "nomic-embed", "mxbai-embed")


@dataclass
class EvalJob:
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _BackendState:
    fingerprint: str | None = None
    agent: ReActAgent | None = None
    memory: ChatMemoryBuffer | None = None
    index: Any = None
    query_engine_cache: dict[str, Any] = field(default_factory=dict)


_backend = _BackendState()
_eval_jobs: dict[str, EvalJob] = {}
_eval_lock = threading.Lock()


def apply_grounding_mode(mode: GroundingMode) -> None:
    if mode == "strict":
        settings.grounding_strictness = "strict"
        settings.response_prompt_version = "v2_strict"
        settings.faithfulness_guard_mode = "strict"
        settings.strict_grounding = True
    else:
        settings.grounding_strictness = "balanced"
        settings.response_prompt_version = "v2_balanced"
        settings.faithfulness_guard_mode = "balanced"
        settings.strict_grounding = False


def apply_llm_model(model: str) -> None:
    """Switch active Ollama LLM at runtime."""
    settings.llm_model = model
    configure_llama_index()
    Settings.llm = configure_llm()
    _backend.fingerprint = None
    _backend.query_engine_cache.clear()
    logger.info("LLM model switched to %s", model)


def list_available_models() -> dict[str, Any]:
    ok, names, err = probe_ollama_tags()
    chat_models = filter_chat_models(names)
    if settings.llm_model not in chat_models:
        chat_models = [settings.llm_model] + chat_models
    return {
        "connected": ok,
        "error": err,
        "active_model": settings.llm_model,
        "models": [
            {"id": m, "label": format_model_label(m)} for m in chat_models
        ],
    }


def get_system_health() -> dict[str, Any]:
    probe = probe_chroma_index()
    stats = get_collection_stats()
    ollama_ok, _, ollama_err = probe_ollama_tags()
    chroma_db = settings.chroma_persist_dir / "chroma.sqlite3"
    last_updated: str | None = None
    if chroma_db.exists():
        mtime = datetime.fromtimestamp(chroma_db.stat().st_mtime, tz=timezone.utc)
        last_updated = mtime.strftime("%Y-%m-%d %H:%M UTC")
    return {
        "index_ready": probe.get("ready", False),
        "chunk_count": stats.get("count", 0),
        "collection": stats.get("collection"),
        "last_updated": last_updated,
        "ollama_connected": ollama_ok,
        "ollama_error": ollama_err,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "retrieval": get_retrieval_config_summary(),
    }


def _settings_fingerprint(
    *,
    corpus_scope: CorpusScope,
    chat_mode: ChatMode,
    llm_model: str,
) -> str:
    return "|".join(
        [
            resolve_grounding_mode(),
            str(settings.llm_temperature),
            settings.faithfulness_guard_mode,
            str(settings.enable_conversation_memory),
            llm_model,
            corpus_scope,
            chat_mode,
        ]
    )


def _ensure_backend(
    *,
    corpus_scope: CorpusScope = "all",
    chat_mode: ChatMode = "direct",
    llm_model: str | None = None,
) -> None:
    if llm_model and llm_model != settings.llm_model:
        apply_llm_model(llm_model)

    fingerprint = _settings_fingerprint(
        corpus_scope=corpus_scope,
        chat_mode=chat_mode,
        llm_model=settings.llm_model,
    )
    if _backend.fingerprint == fingerprint and _backend.agent is not None:
        return

    configure_llama_index()
    Settings.llm = configure_llm()
    index = get_or_create_index()
    filters = corpus_retrieval_filters(corpus_scope) if corpus_scope != "all" else None
    memory = _backend.memory or create_session_memory()
    agent = create_agent(index, memory=memory, scope_filters=filters)
    _backend.agent = agent
    _backend.memory = memory
    _backend.index = index
    _backend.query_engine_cache.clear()
    _backend.fingerprint = fingerprint
    logger.info(
        "Chat backend initialized | model=%s scope=%s mode=%s",
        settings.llm_model,
        corpus_scope,
        chat_mode,
    )


def _extract_query_answer(response: Any) -> str:
    if response is None:
        return "I could not generate a response."
    inner = getattr(response, "response", response)
    if inner is None:
        return "I could not generate a response."
    text = getattr(inner, "response", None) or getattr(inner, "text", None) or inner
    return str(text).strip()


def _get_query_engine(
    user_message: str,
    corpus_scope: CorpusScope,
) -> Any:
    filters = resolve_query_filters(user_message, corpus_scope)
    cache_key = f"{_backend.fingerprint}|{filters!r}"
    if cache_key in _backend.query_engine_cache:
        return _backend.query_engine_cache[cache_key]

    index = _backend.index or get_or_create_index()
    engine = build_query_engine(index, filters=filters)
    _backend.query_engine_cache[cache_key] = engine
    return engine


def _run_direct_turn(user_message: str, corpus_scope: CorpusScope) -> AgentTurnResult:
    query_engine = _get_query_engine(user_message, corpus_scope)
    begin_query_timing()
    t0 = time.perf_counter()
    try:
        begin_citation_turn()
        response = query_engine.query(append_language_hint(user_message))
        answer = _extract_query_answer(response)
        citations: list[dict[str, Any]] = []
        if settings.show_citations:
            generation_nodes = get_generation_nodes_this_turn()
            citations = select_citations_for_answer(
                answer,
                generation_nodes,
                user_query=user_message,
            )
        record_stage("e2e", (time.perf_counter() - t0) * 1000)
        timing = get_current_timing()
        return AgentTurnResult(
            answer=answer,
            citations=citations,
            timing=timing.as_dict() if timing else None,
            low_confidence=LOW_CONFIDENCE_MESSAGE in answer,
            grounding_mode=resolve_grounding_mode(),
        )
    finally:
        clear_timing()


def _run_agent_turn(user_message: str, corpus_scope: CorpusScope) -> AgentTurnResult:
    if corpus_scope == "all" and resolve_query_filters(user_message, corpus_scope):
        return _run_direct_turn(user_message, corpus_scope)

    agent = _backend.agent
    memory = _backend.memory
    if agent is None:
        raise RuntimeError("Agent not initialized")

    begin_query_timing()
    t0 = time.perf_counter()
    try:
        try:
            turn = asyncio.run(chat_with_memory(agent, user_message, memory=memory))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                turn = loop.run_until_complete(
                    chat_with_memory(agent, user_message, memory=memory)
                )
            finally:
                loop.close()
        record_stage("e2e", (time.perf_counter() - t0) * 1000)
        timing = get_current_timing()
        if timing and turn.timing is None:
            turn.timing = timing.as_dict()
        return turn
    finally:
        clear_timing()


def run_chat_turn(
    message: str,
    *,
    corpus_scope: CorpusScope = "all",
    chat_mode: ChatMode = "direct",
    llm_model: str | None = None,
    grounding_mode: GroundingMode | None = None,
) -> AgentTurnResult:
    """Execute one RAG chat turn (direct or agent mode)."""
    if grounding_mode:
        apply_grounding_mode(grounding_mode)

    probe = probe_chroma_index()
    if not probe.get("ready"):
        raise RuntimeError(
            "Search index is not ready. Index documents before chatting."
        )

    _ensure_backend(
        corpus_scope=corpus_scope,
        chat_mode=chat_mode,
        llm_model=llm_model,
    )

    if chat_mode == "agent":
        return _run_agent_turn(message, corpus_scope)
    return _run_direct_turn(message, corpus_scope)


def turn_to_dict(turn: AgentTurnResult) -> dict[str, Any]:
    return {
        "answer": turn.answer,
        "citations": turn.citations,
        "timing": turn.timing,
        "low_confidence": turn.low_confidence,
        "grounding_mode": turn.grounding_mode,
    }


def _run_eval_job(job_id: str, *, max_samples: int | None) -> None:
    with _eval_lock:
        job = _eval_jobs.get(job_id)
        if job:
            job.status = "running"

    try:
        from src.evaluation import run_evaluation, save_eval_results

        eval_run = run_evaluation(max_samples=max_samples, use_llm_judge=True)
        save_eval_results(eval_run)
        with _eval_lock:
            job = _eval_jobs.get(job_id)
            if job:
                job.status = "completed"
                job.result = {
                    "run_id": eval_run.run_id,
                    "aggregate": eval_run.aggregate,
                }
    except Exception as exc:
        logger.exception("Eval job %s failed", job_id)
        with _eval_lock:
            job = _eval_jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = str(exc)


def start_eval_job(*, max_samples: int | None = 5) -> EvalJob:
    job_id = str(uuid.uuid4())[:8]
    job = EvalJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with _eval_lock:
        _eval_jobs[job_id] = job

    thread = threading.Thread(
        target=_run_eval_job,
        args=(job_id,),
        kwargs={"max_samples": max_samples},
        daemon=True,
    )
    thread.start()
    return job


def get_eval_job(job_id: str) -> EvalJob | None:
    with _eval_lock:
        return _eval_jobs.get(job_id)