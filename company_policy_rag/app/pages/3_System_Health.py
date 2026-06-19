"""Admin system health and diagnostics page."""

from __future__ import annotations

import streamlit as st

from app.ui.bootstrap import bootstrap_app
from app.ui.components.health import (
    format_eval_metrics,
    load_last_eval_run,
    probe_ollama_tags,
)
from app.ui.session import ensure_session_state, index_health
from app.ui.sidebar import render_sidebar_controls, render_sidebar_status
from app.ui.theme import inject_global_styles
from src.config import settings
from src.indexing import probe_chroma_index, reset_chroma_client_cache
from src.retriever import get_retrieval_config_summary
from src.timing import summarize_ms

bootstrap_app()
inject_global_styles()
ensure_session_state()
render_sidebar_controls()
render_sidebar_status()

st.title("System health")
st.caption("Chroma index, Ollama models, retrieval config, and latest eval snapshot.")

st.subheader("Chroma index")
probe = probe_chroma_index()
cols = st.columns(4)
cols[0].metric("Ready", "Yes" if probe["ready"] else "No")
cols[1].metric("Chunks", probe.get("count", 0))
cols[2].metric("Collection", probe.get("collection", "—"))
health = index_health()
if health.get("last_updated"):
    cols[3].metric("Last updated", health["last_updated"])

with st.expander("Chroma diagnostics", expanded=not probe["ready"]):
    st.code(
        "\n".join(
            [
                f"Persist dir: {probe['persist_dir']}",
                f"Dir exists: {probe['dir_exists']}",
                f"SQLite exists: {probe['sqlite_exists']}",
                f"Collections: {', '.join(probe['collections']) or 'none'}",
                f"Error: {probe.get('error') or 'none'}",
            ]
        ),
        language="text",
    )
    if st.button("Clear Chroma client cache", key="health_clear_chroma"):
        reset_chroma_client_cache()
        st.rerun()

st.subheader("Ollama")
ok, models, err = probe_ollama_tags(settings.ollama_base_url)
if ok:
    st.success(f"Connected to `{settings.ollama_base_url}`")
    st.caption(f"LLM: `{settings.llm_model}` · Embeddings: `{settings.embed_model}`")
    if models:
        st.markdown("**Installed models**")
        for name in models:
            st.text(f"• {name}")
    else:
        st.warning("Ollama responded but no models are installed.")
else:
    st.error(f"Cannot reach Ollama at `{settings.ollama_base_url}`: {err}")

st.subheader("Retrieval config")
st.json(get_retrieval_config_summary())

st.subheader("Corpus scope")
scope = st.session_state.get("corpus_scope", "all")
st.caption(f"Active chat scope: **{scope}** (change on Chat page sidebar)")

st.subheader("Session latency")
samples = st.session_state.get("timing_samples") or []
summary = summarize_ms(samples)
if summary["count"]:
    mcols = st.columns(4)
    mcols[0].metric("Queries", int(summary["count"]))
    mcols[1].metric("Mean E2E", f"{summary['mean']:.0f} ms")
    mcols[2].metric("p50 E2E", f"{summary['p50']:.0f} ms")
    mcols[3].metric("p95 E2E", f"{summary['p95']:.0f} ms")
else:
    st.caption("No timing samples yet — run a chat query on the Chat page.")

st.subheader("Latest evaluation")
eval_path = settings.eval_results_path
last_run = load_last_eval_run(eval_path)
if last_run:
    st.caption(f"Run ID: `{last_run.get('run_id', '—')}` · Source: `{eval_path}`")
    metrics = format_eval_metrics(last_run)
    if metrics:
        ecols = st.columns(len(metrics))
        for col, (label, value) in zip(ecols, metrics.items()):
            col.metric(label, value)
    else:
        st.caption("Run found but no aggregate metrics recorded.")
else:
    st.info(
        f"No eval runs at `{eval_path}`. Run `python scripts/evaluate.py` from project root."
    )