"""Employee chat page."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_ensure_path",
    Path(__file__).resolve().parents[1] / "_ensure_path.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod.ensure_project_root()

import streamlit as st

from app.ui.bootstrap import bootstrap_app
from app.ui.components.chat import render_chat_interface
from app.ui.components.citations import render_grounding_badge
from app.ui.session import ensure_backend_ready, ensure_query_engine, ensure_session_state
from app.ui.sidebar import render_sidebar_controls, render_sidebar_status
from app.ui.theme import inject_global_styles
from src.indexing import probe_chroma_index, reset_chroma_client_cache

bootstrap_app()
inject_global_styles()
ensure_session_state()
render_sidebar_controls()

in_conversation = bool(st.session_state.get("messages"))

if not in_conversation:
    st.title("Company Policy Assistant")
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.caption("Answers grounded in indexed policy and legal documents.")
    with header_cols[1]:
        render_grounding_badge()

probe = probe_chroma_index()
if not probe["ready"]:
    st.warning(
        "**No searchable index yet.** Open **Documents** to upload a legal PDF, "
        "or place handbook PDFs in `data/policies/` and run indexing."
    )
    with st.expander("Index diagnostics", expanded=False):
        st.code(
            "\n".join(
                [
                    f"Chroma dir: {probe['persist_dir']}",
                    f"Dir exists: {probe['dir_exists']}",
                    f"SQLite exists: {probe['sqlite_exists']}",
                    f"Collections: {', '.join(probe['collections']) or 'none'}",
                    f"Target collection: {probe['collection']}",
                    f"Chunks: {probe['count']}",
                    f"Error: {probe.get('error') or 'none'}",
                ]
            ),
            language="text",
        )
        if st.button("Clear Chroma cache and retry", key="chat_retry_chroma"):
            reset_chroma_client_cache()
            st.rerun()
    render_sidebar_status()
    st.stop()

if not ensure_backend_ready():
    render_sidebar_status()
    st.stop()

render_sidebar_status()

agent = st.session_state.get("agent")
memory = st.session_state.get("memory")
query_engine = ensure_query_engine()

st.markdown('<div class="chat-page">', unsafe_allow_html=True)
render_chat_interface(agent, memory, query_engine)
st.markdown("</div>", unsafe_allow_html=True)