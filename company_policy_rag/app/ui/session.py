"""Streamlit session state and RAG backend lifecycle."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from src.agent import AgentTurnResult, chat_with_memory, configure_llm, create_agent
from src.citations import (
    begin_citation_turn,
    get_generation_nodes_this_turn,
    select_citations_for_answer,
)
from src.config import settings
from src.language import append_language_hint
from src.prompts import LOW_CONFIDENCE_MESSAGE, resolve_grounding_mode
from src.indexing import (
    configure_llama_index,
    get_collection_stats,
    get_or_create_index,
    probe_chroma_index,
    reset_chroma_client_cache,
)
from src.memory import create_session_memory
from src.prompts import resolve_grounding_mode
from src.retrieval_scope import corpus_retrieval_filters, resolve_query_filters
from src.retriever import build_query_engine, build_retriever
from src.timing import begin_query_timing, clear_timing, get_current_timing, record_stage
from src.utils import logger


def apply_grounding_mode(mode: str) -> None:
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


def corpus_scope_filters(scope: str | None) -> dict[str, str] | None:
    key = (scope or "all").lower().strip()
    if key in ("", "all"):
        return None
    return corpus_retrieval_filters(key)


def settings_fingerprint() -> str:
    scope = st.session_state.get("corpus_scope", "all")
    chat_mode = st.session_state.get("chat_mode", "direct")
    return "|".join(
        [
            resolve_grounding_mode(),
            str(settings.llm_temperature),
            settings.faithfulness_guard_mode,
            str(settings.enable_conversation_memory),
            str(settings.show_citations),
            str(settings.citation_show_excerpts),
            str(settings.citation_show_relevance_score),
            scope,
            chat_mode,
        ]
    )


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "corpus_scope" not in st.session_state:
        st.session_state.corpus_scope = "all"
    if "timing_samples" not in st.session_state:
        st.session_state.timing_samples = []
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "direct"
    if "pending_user_prompt" not in st.session_state:
        st.session_state.pending_user_prompt = None


def initialize_backend(
    memory: ChatMemoryBuffer | None = None,
    *,
    scope_filters: dict[str, str] | None = None,
) -> tuple[ReActAgent, Any, ChatMemoryBuffer | None, Any]:
    configure_llama_index()
    from llama_index.core import Settings

    Settings.llm = configure_llm()
    index = get_or_create_index()
    session_memory = memory if memory is not None else create_session_memory()
    agent = create_agent(index, memory=session_memory, scope_filters=scope_filters)
    retriever = build_retriever(index, filters=scope_filters)
    return agent, retriever, session_memory, index


def reload_rag_session() -> None:
    reset_chroma_client_cache()
    st.session_state.initialized = False
    st.session_state.pop("agent", None)
    st.session_state.pop("retriever", None)
    st.session_state.pop("query_engine", None)
    st.session_state.pop("settings_fingerprint", None)


def _extract_query_answer(response: Any) -> str:
    if response is None:
        return "I could not generate a response."
    inner = getattr(response, "response", response)
    if inner is None:
        return "I could not generate a response."
    text = getattr(inner, "response", None) or getattr(inner, "text", None) or inner
    return str(text).strip()


def ensure_query_engine(user_message: str | None = None) -> Any:
    """Cached query engine; per-query corpus routing when scope is 'all'."""
    scope = st.session_state.get("corpus_scope", "all")
    if user_message:
        filters = resolve_query_filters(user_message, scope)
    else:
        filters = corpus_scope_filters(scope)
    fingerprint = settings_fingerprint()
    cache_key = f"{fingerprint}|{filters!r}"
    if st.session_state.get("query_engine_cache_key") == cache_key:
        cached = st.session_state.get("query_engine")
        if cached is not None:
            return cached

    index = get_or_create_index()
    engine = build_query_engine(index, filters=filters)
    st.session_state.query_engine = engine
    st.session_state.query_engine_cache_key = cache_key
    return engine


def run_direct_turn(user_message: str) -> AgentTurnResult:
    """Single-shot RAG without ReAct agent overhead."""
    query_engine = ensure_query_engine(user_message)
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
        timing_dict = timing.as_dict() if timing else None
        if timing_dict:
            samples: list[float] = st.session_state.get("timing_samples", [])
            e2e = timing_dict.get("e2e_ms", 0)
            if e2e:
                samples.append(float(e2e))
                st.session_state.timing_samples = samples[-50:]

        return AgentTurnResult(
            answer=answer,
            citations=citations,
            timing=timing_dict,
            low_confidence=LOW_CONFIDENCE_MESSAGE in answer,
            grounding_mode=resolve_grounding_mode(),
        )
    finally:
        clear_timing()


def run_agent_turn(
    agent: ReActAgent,
    user_message: str,
    memory: ChatMemoryBuffer | None,
) -> AgentTurnResult:
    scope = st.session_state.get("corpus_scope", "all")
    if scope == "all" and resolve_query_filters(user_message, scope):
        return run_direct_turn(user_message)

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
        if turn.timing:
            samples: list[float] = st.session_state.get("timing_samples", [])
            e2e = turn.timing.get("e2e_ms", 0)
            if e2e:
                samples.append(float(e2e))
                st.session_state.timing_samples = samples[-50:]
        return turn
    finally:
        clear_timing()


def index_health() -> dict[str, Any]:
    stats = get_collection_stats()
    chroma_db = settings.chroma_persist_dir / "chroma.sqlite3"
    last_updated: str | None = None
    if chroma_db.exists():
        mtime = datetime.fromtimestamp(chroma_db.stat().st_mtime, tz=timezone.utc)
        last_updated = mtime.strftime("%Y-%m-%d %H:%M UTC")
    return {**stats, "last_updated": last_updated}


def ensure_backend_ready(*, require_index: bool = True) -> bool:
    """
    Initialize agent if settings changed. Returns False if index missing or init failed.
    """
    ensure_session_state()
    if require_index:
        probe = probe_chroma_index()
        if not probe["ready"]:
            return False

    fingerprint = settings_fingerprint()
    needs_init = (
        not st.session_state.initialized
        or st.session_state.get("settings_fingerprint") != fingerprint
    )

    if needs_init:
        try:
            scope = st.session_state.get("corpus_scope", "all")
            filters = corpus_scope_filters(scope)
            preserved_memory = st.session_state.get("memory")
            agent, retriever, memory, _index = initialize_backend(
                memory=preserved_memory,
                scope_filters=filters,
            )
            st.session_state.agent = agent
            st.session_state.retriever = retriever
            st.session_state.memory = memory
            st.session_state.query_engine = build_query_engine(_index, filters=filters)
            st.session_state.query_engine_fingerprint = fingerprint
            st.session_state.initialized = True
            st.session_state.settings_fingerprint = fingerprint
            logger.info(
                "Streamlit session initialized | grounding=%s corpus=%s",
                resolve_grounding_mode(),
                scope,
            )
        except Exception as exc:
            logger.exception("Streamlit backend initialization failed")
            st.error(f"Failed to initialize RAG backend: {exc}")
            return False
    return True