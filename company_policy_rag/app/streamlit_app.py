"""
Streamlit chat interface for the Company Policy RAG system.

Run from the project root:
    streamlit run app/streamlit_app.py

Citation UX (production-rag: trust through verifiable sources):
- Assistant answers render in the main chat thread.
- Expandable source cards below each answer show section_path, page, file, and excerpt.
"""

from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when Streamlit loads this module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from src.agent import AgentTurnResult, chat_with_memory, create_agent, configure_llm
from src.config import settings
from src.document_upload import (
    RemoveDocumentResult,
    index_legal_paths,
    list_legal_documents,
    remove_legal_document,
    save_legal_pdf,
)
from src.indexing import (
    IndexingResult,
    configure_llama_index,
    get_collection_stats,
    get_or_create_index,
    probe_chroma_index,
    reset_chroma_client_cache,
)
from src.memory import create_session_memory, memory_stats
from src.prompts import resolve_grounding_mode
from src.retriever import build_retriever
from src.utils import (
    format_citation_excerpt,
    format_citation_primary,
    logger,
    prepare_citations_for_display,
    setup_logging,
    shorten_source_filename,
)

setup_logging("streamlit")

# Drop any stale Chroma client from a prior Streamlit hot-reload (settings mismatch).
reset_chroma_client_cache()

# ── Page config & styling ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
    /* Tighter chat spacing; subtle assistant source block */
    [data-testid="stChatMessage"] { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .source-header {
        font-size: 0.85rem;
        color: #5f6368;
        margin: 0.75rem 0 0.35rem 0;
        padding-top: 0.75rem;
        border-top: 1px solid #e8eaed;
    }
    .source-meta { font-size: 0.8rem; color: #80868b; margin-bottom: 0.25rem; }
    div[data-testid="stExpander"] details summary p { font-weight: 500; }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #e8f0fe;
        color: #1967d2;
    }
    .status-pill.strict { background: #fce8e6; color: #c5221f; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ── Session helpers ──────────────────────────────────────────────────────────


def _settings_fingerprint() -> str:
    """Detect sidebar changes that require agent re-initialization."""
    return "|".join(
        [
            resolve_grounding_mode(),
            str(settings.llm_temperature),
            settings.faithfulness_guard_mode,
            str(settings.enable_conversation_memory),
        ]
    )


def _apply_grounding_mode(mode: str) -> None:
    """Map sidebar selection to generation / guard settings."""
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


def _initialize_backend(
    memory: ChatMemoryBuffer | None = None,
) -> tuple[ReActAgent, Any, ChatMemoryBuffer | None, Any]:
    """Load index, agent, and retriever — one bundle per Streamlit session."""
    configure_llama_index()
    from llama_index.core import Settings

    Settings.llm = configure_llm()
    index = get_or_create_index()
    session_memory = memory if memory is not None else create_session_memory()
    agent = create_agent(index, memory=session_memory)
    retriever = build_retriever(index)
    return agent, retriever, session_memory, index


def _ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def _run_agent_turn(
    agent: ReActAgent,
    user_message: str,
    memory: ChatMemoryBuffer | None,
) -> AgentTurnResult:
    """Async agent chat from Streamlit's sync context."""
    try:
        return asyncio.run(chat_with_memory(agent, user_message, memory=memory))
    except RuntimeError:
        # Fallback when an event loop is already running (some Streamlit hosts).
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                chat_with_memory(agent, user_message, memory=memory)
            )
        finally:
            loop.close()


def _reload_rag_session() -> None:
    """Force agent re-init after the Chroma index changes."""
    reset_chroma_client_cache()
    st.session_state.initialized = False
    st.session_state.pop("agent", None)
    st.session_state.pop("retriever", None)
    st.session_state.pop("settings_fingerprint", None)


def _format_indexing_result(result: IndexingResult) -> str:
    lines = [
        f"PDFs processed: {', '.join(result.pdf_files_processed) or 'none'}",
        f"Chunks inserted: {result.nodes_inserted}",
        f"Documents loaded: {result.documents_loaded}",
    ]
    if result.pdf_files_skipped:
        lines.append(f"Skipped (unchanged): {', '.join(result.pdf_files_skipped)}")
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"  - {err}" for err in result.errors)
    return "\n".join(lines)


def _index_changed(result: IndexingResult | None) -> bool:
    if result is None:
        return False
    return result.nodes_inserted > 0 or bool(result.pdf_files_processed)


def _show_upload_outcome(result: IndexingResult | None, error: str | None, *, compact: bool = False) -> bool:
    """
    Display upload/index outcome. Returns True if the RAG session should reload.
    """
    if error:
        st.error(error)
        return False

    if result is None:
        return False

    if _index_changed(result):
        stats = get_collection_stats()
        st.success(
            f"Indexed successfully. Total chunks in collection: {stats.get('count', 0)}"
        )
        body = _format_indexing_result(result)
        if compact:
            st.caption(body)
        else:
            st.code(body, language="text")
        return True

    # Saved to disk but content already in Chroma (common when handbook exists in policies/).
    st.info(
        "File saved to `data/legal/`, but content is **already indexed** "
        "(unchanged hash). Chat is ready — no re-index needed."
    )
    if compact:
        st.caption(_format_indexing_result(result))
    else:
        st.code(_format_indexing_result(result), language="text")
    return False


def _process_legal_removal(filename: str) -> tuple[RemoveDocumentResult | None, str | None]:
    """Remove a legal PDF from disk and the search index."""
    try:
        result = remove_legal_document(filename)
        return result, None
    except FileNotFoundError as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:
        logger.exception("Legal document removal failed")
        return None, f"Removal failed: {exc}"


def _process_legal_uploads(uploaded_files: list[Any]) -> tuple[list[Path], IndexingResult | None, str | None]:
    """Save uploaded PDFs and run incremental indexing. Returns paths, result, error."""
    if not uploaded_files:
        return [], None, "Select at least one PDF to upload."

    saved_paths: list[Path] = []
    try:
        for uploaded in uploaded_files:
            saved_paths.append(save_legal_pdf(uploaded.getvalue(), uploaded.name))
        with st.spinner("Indexing legal documents (embeddings via Ollama)…"):
            result = index_legal_paths(saved_paths)
        return saved_paths, result, None
    except Exception as exc:
        logger.exception("Legal document upload/index failed")
        return saved_paths, None, str(exc)


def _index_health() -> dict[str, Any]:
    """Chunk count plus approximate last-index time from Chroma persistence."""
    stats = get_collection_stats()
    chroma_db = settings.chroma_persist_dir / "chroma.sqlite3"
    last_updated: str | None = None
    if chroma_db.exists():
        mtime = datetime.fromtimestamp(chroma_db.stat().st_mtime, tz=timezone.utc)
        last_updated = mtime.strftime("%Y-%m-%d %H:%M UTC")
    return {**stats, "last_updated": last_updated}


# ── Citation rendering ───────────────────────────────────────────────────────


def _render_sources(citations: list[dict[str, Any]]) -> None:
    """Expandable source cards — section, page, file, excerpt."""
    prepared = prepare_citations_for_display(citations)
    if not prepared:
        return

    noun = "source" if len(prepared) == 1 else "sources"
    st.markdown(
        f'<p class="source-header">📚 {len(prepared)} {noun} — expand to verify against the policy document</p>',
        unsafe_allow_html=True,
    )

    for i, citation in enumerate(prepared, 1):
        label = format_citation_primary(citation)
        if settings.citation_show_relevance_score and citation.get("score") is not None:
            label = f"{label} · score {citation['score']:.2f}"

        with st.expander(f"{i}. {label}", expanded=False):
            reason = citation.get("selection_reason")
            if reason == "cited_in_answer":
                st.caption("Cited in answer")
            elif reason == "score_threshold_fallback":
                st.caption("High relevance to question (no explicit [Source N] tag in answer)")

            source_file = citation.get("source_file", "unknown")
            st.markdown(f"**Document:** `{shorten_source_filename(source_file)}`")
            st.markdown(f"**File:** `{source_file}`")

            section_path = citation.get("section_path")
            if section_path:
                st.markdown(f"**Section path:** {section_path}")
            elif citation.get("section_title"):
                st.markdown(f"**Section:** {citation.get('section_title')}")

            if citation.get("section_number"):
                st.markdown(f"**Section #:** {citation['section_number']}")

            page = citation.get("page_number")
            if page is not None:
                st.markdown(f"**Page:** {page}")

            if settings.citation_show_excerpts:
                st.markdown("**Excerpt**")
                st.caption(format_citation_excerpt(citation, max_len=500))


def _render_grounding_badge() -> None:
    mode = resolve_grounding_mode()
    css_class = "status-pill strict" if mode == "strict" else "status-pill"
    label = "Strict grounding" if mode == "strict" else "Balanced grounding"
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────


def _render_sidebar_controls() -> None:
    """Settings widgets — render early so grounding/temperature apply before init."""
    with st.sidebar:
        st.title("Settings")
        st.caption("Internal policy & legal document assistant")

        grounding_choice = st.radio(
            "Grounding mode",
            options=["balanced", "strict"],
            format_func=lambda x: "Balanced (helpful synthesis)" if x == "balanced" else "Strict (max faithfulness)",
            index=0 if resolve_grounding_mode() == "balanced" else 1,
            help="Balanced answers from related excerpts; Strict abstains unless fully supported.",
        )
        _apply_grounding_mode(grounding_choice)

        settings.llm_temperature = st.slider(
            "LLM temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.llm_temperature),
            step=0.05,
            help="Lower values produce more deterministic, factual responses.",
        )

        st.divider()
        st.subheader("Session")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            memory = st.session_state.get("memory")
            if memory is not None:
                memory.reset()
            st.rerun()


def _render_sidebar_status() -> None:
    """System status + legal upload — render after backend init in main()."""
    with st.sidebar:
        st.divider()
        st.subheader("System status")
        health = _index_health()
        if health.get("ready"):
            st.metric("Indexed chunks", health.get("count", 0))
            if health.get("last_updated"):
                st.caption(f"Index last updated: {health['last_updated']}")
            st.caption(f"Collection: `{health.get('collection', '—')}`")
            if st.session_state.get("initialized"):
                st.caption(f"LLM: `{settings.llm_model}`")
                st.caption(f"Embeddings: `{settings.embed_model}`")
                if settings.enable_reranker:
                    st.caption(
                        f"Retrieval: {settings.retrieval_candidate_k} candidates → "
                        f"rerank → top {settings.reranker_top_n}"
                    )
                mem = memory_stats(st.session_state.get("memory"))
                if mem.get("enabled"):
                    st.caption(f"Memory: {mem.get('turns', 0)} turns in session")
            else:
                st.caption("Agent loading…")
        else:
            st.warning("No index yet — upload a legal PDF in the main panel.")

        st.divider()
        _render_legal_upload_sidebar()


def _render_legal_file_list(*, key_prefix: str, compact: bool = False) -> list[dict]:
    """Show uploaded legal PDFs with per-file Remove buttons."""
    docs = list_legal_documents()
    if not docs:
        return docs

    st.markdown("**Uploaded legal PDFs**")
    for index, doc in enumerate(docs):
        cols = st.columns([5, 1] if compact else [6, 1])
        label = f"{doc['filename']} ({doc['size_kb']} KB)"
        with cols[0]:
            if compact:
                st.caption(label)
            else:
                st.text(f"{label} — updated {doc['modified_utc']}")
        with cols[1]:
            if st.button(
                "Remove",
                key=f"{key_prefix}_remove_{index}",
                help=f"Delete {doc['filename']} from disk and the search index",
            ):
                removal, error = _process_legal_removal(doc["filename"])
                if error:
                    st.error(error)
                elif removal is not None:
                    stats = get_collection_stats()
                    st.success(
                        f"Removed `{removal.filename}` "
                        f"({removal.chunks_removed} chunks deleted). "
                        f"Collection now has {stats.get('count', 0)} chunks."
                    )
                    _reload_rag_session()
                    st.rerun()
    return docs


def _render_legal_upload_sidebar() -> None:
    """Compact legal PDF uploader in the sidebar."""
    st.subheader("Legal documents")
    st.caption("Upload PDFs to `data/legal/` and index them for search.")

    docs = _render_legal_file_list(key_prefix="legal_sidebar", compact=True)
    if not docs:
        st.caption("No legal PDFs uploaded yet.")

    uploads = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="legal_upload_sidebar",
        label_visibility="collapsed",
    )

    if st.button("Save & index", key="legal_index_sidebar", use_container_width=True, disabled=not uploads):
        _, result, error = _process_legal_uploads(uploads or [])
        if _show_upload_outcome(result, error, compact=True):
            _reload_rag_session()
            st.rerun()


# ── Legal documents panel ─────────────────────────────────────────────────────


def _render_legal_documents_panel(*, expanded: bool = False) -> None:
    """Full legal document manager on the main page."""
    with st.expander("Manage legal documents", expanded=expanded):
        st.caption(
            "Upload legal PDFs here. Files are saved under `data/legal/` and embedded into the search index. "
            "Use **Remove** to delete a PDF you no longer want in search results."
        )

        docs = _render_legal_file_list(key_prefix="legal_main", compact=False)
        if not docs:
            st.info("No legal PDFs uploaded yet.")

        uploads = st.file_uploader(
            "Upload legal PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="legal_upload_main",
        )

        if st.button("Save & index", key="legal_index_main", type="primary", disabled=not uploads):
            _, result, error = _process_legal_uploads(uploads or [])
            if _show_upload_outcome(result, error):
                _reload_rag_session()
                st.rerun()


# ── Chat UI ──────────────────────────────────────────────────────────────────


def _render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations") and settings.show_citations:
                _render_sources(msg["citations"])


def _render_welcome() -> None:
    if st.session_state.messages:
        return

    health = _index_health()
    st.info(
        "Ask about employee handbook policies, benefits, leave, conduct rules, or legal documents. "
        "Follow-up questions work — recent conversation is remembered in this session."
    )
    cols = st.columns(3)
    cols[0].metric("Indexed chunks", health.get("count", 0))
    cols[1].metric("Grounding", resolve_grounding_mode().title())
    cols[2].metric("Citations", "On" if settings.show_citations else "Off")


def main() -> None:
    _ensure_session_state()
    _render_sidebar_controls()

    st.title("Company Policy Assistant")
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.caption("Answers are grounded in your indexed policy and legal documents.")
    with header_cols[1]:
        _render_grounding_badge()

    probe = probe_chroma_index()
    index_ready = probe["ready"]

    if not index_ready:
        st.warning(
            "**No searchable index yet.** Upload a legal PDF below to create one, "
            "or place handbook PDFs in `data/policies/` and run indexing."
        )
        _render_legal_documents_panel(expanded=True)
        with st.expander("Index diagnostics & CLI fallback", expanded=False):
            st.code(
                "\n".join(
                    [
                        f"Project root: {PROJECT_ROOT}",
                        f"Chroma dir: {probe['persist_dir']}",
                        f"Dir exists: {probe['dir_exists']}",
                        f"SQLite exists: {probe['sqlite_exists']}",
                        f"Collections: {', '.join(probe['collections']) or 'none'}",
                        f"Target collection: {probe['collection']}",
                        f"Chunks (actual): {probe['count']}",
                        f"Error: {probe.get('error') or 'none'}",
                    ]
                ),
                language="text",
            )
            st.markdown(
                "Handbook / bulk indexing:\n\n```bash\npython scripts/index_documents.py\n```"
            )
            if st.button("Clear Chroma client cache and retry", key="retry_chroma_cache"):
                reset_chroma_client_cache()
                st.rerun()
        probe = probe_chroma_index()
        index_ready = probe["ready"]
        if not index_ready:
            _render_sidebar_status()
            return

    fingerprint = _settings_fingerprint()
    needs_init = (
        not st.session_state.initialized
        or st.session_state.get("settings_fingerprint") != fingerprint
    )

    if needs_init:
        try:
            with st.spinner("Loading policy assistant (index + agent)…"):
                preserved_memory = st.session_state.get("memory")
                agent, retriever, memory, _index = _initialize_backend(memory=preserved_memory)
            st.session_state.agent = agent
            st.session_state.retriever = retriever
            st.session_state.memory = memory
            st.session_state.initialized = True
            st.session_state.settings_fingerprint = fingerprint
            logger.info("Streamlit session initialized | grounding=%s", resolve_grounding_mode())
        except Exception as exc:
            logger.exception("Streamlit backend initialization failed")
            st.error(f"Failed to initialize RAG backend: {exc}")
            _render_sidebar_status()
            return

    _render_sidebar_status()

    agent: ReActAgent | None = st.session_state.get("agent")
    retriever = st.session_state.get("retriever")
    memory = st.session_state.get("memory")

    _render_welcome()
    _render_chat_history()

    # Legal upload below chat history; chat_input must stay last for sticky footer.
    _render_legal_documents_panel(expanded=False)

    if prompt := st.chat_input("Ask a policy question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Searching policies and generating answer…", expanded=False):
                try:
                    st.write("Running policy search and synthesizing grounded answer…")
                    turn = _run_agent_turn(agent, prompt, memory)
                    answer = turn.answer
                    citations = turn.citations
                except Exception as exc:
                    logger.exception("Agent query failed")
                    answer = f"Sorry, something went wrong: {exc}"
                    citations = []

            st.markdown(answer)
            if settings.show_citations and citations:
                _render_sources(citations)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )


if __name__ == "__main__":
    main()