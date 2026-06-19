"""Legal PDF upload and handbook indexing helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import settings
from src.document_upload import (
    RemoveDocumentResult,
    index_legal_paths,
    list_legal_documents,
    remove_legal_document,
    save_legal_pdf,
)
from src.indexing import IndexingResult, get_collection_stats
from src.utils import logger

from app.ui.bootstrap import PROJECT_ROOT
from app.ui.session import reload_rag_session


def format_indexing_result(result: IndexingResult) -> str:
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


def index_changed(result: IndexingResult | None) -> bool:
    if result is None:
        return False
    return result.nodes_inserted > 0 or bool(result.pdf_files_processed)


def show_upload_outcome(
    result: IndexingResult | None,
    error: str | None,
) -> bool:
    """Returns True if RAG session should reload."""
    if error:
        st.error(error)
        return False
    if result is None:
        return False
    if index_changed(result):
        stats = get_collection_stats()
        st.success(
            f"Indexed successfully. Total chunks: {stats.get('count', 0)}"
        )
        st.code(format_indexing_result(result), language="text")
        return True
    st.info(
        "File saved but content is **already indexed** (unchanged hash). "
        "No re-index needed."
    )
    st.code(format_indexing_result(result), language="text")
    return False


def process_legal_removal(filename: str) -> tuple[RemoveDocumentResult | None, str | None]:
    try:
        return remove_legal_document(filename), None
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)
    except Exception as exc:
        logger.exception("Legal document removal failed")
        return None, f"Removal failed: {exc}"


def process_legal_uploads(
    uploaded_files: list[Any],
) -> tuple[list[Path], IndexingResult | None, str | None]:
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


def list_handbook_pdfs() -> list[Path]:
    policies_dir = settings.policies_dir
    if not policies_dir.exists():
        return []
    return sorted(policies_dir.glob("**/*.pdf"))


def run_handbook_indexing() -> tuple[bool, str]:
    """Run scripts/index_documents.py; returns (success, output)."""
    script = PROJECT_ROOT / "scripts" / "index_documents.py"
    if not script.is_file():
        return False, f"Script not found: {script}"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip() or "(no output)"


def render_legal_file_list(*, key_prefix: str) -> list[dict]:
    docs = list_legal_documents()
    if not docs:
        return docs

    st.markdown("**Uploaded legal PDFs**")
    for index, doc in enumerate(docs):
        cols = st.columns([6, 1])
        label = f"{doc['filename']} ({doc['size_kb']} KB) — {doc['modified_utc']}"
        with cols[0]:
            st.text(label)
        with cols[1]:
            if st.button("Remove", key=f"{key_prefix}_remove_{index}"):
                removal, error = process_legal_removal(doc["filename"])
                if error:
                    st.error(error)
                elif removal is not None:
                    stats = get_collection_stats()
                    st.success(
                        f"Removed `{removal.filename}` ({removal.chunks_removed} chunks). "
                        f"Collection: {stats.get('count', 0)} chunks."
                    )
                    reload_rag_session()
                    st.rerun()
    return docs


def render_documents_page() -> None:
    """Admin page: legal uploads and handbook bulk indexing."""
    st.title("Document management")
    st.caption(
        "Upload legal PDFs to `data/legal/` or re-index handbook PDFs from `data/policies/`."
    )

    st.subheader("Legal documents")
    st.caption(
        "Files are embedded into the search index automatically. "
        "Use **Remove** to delete a PDF from disk and the index."
    )

    docs = render_legal_file_list(key_prefix="legal_page")
    if not docs:
        st.info("No legal PDFs uploaded yet.")

    uploads = st.file_uploader(
        "Upload legal PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="legal_upload_page",
    )
    if st.button(
        "Save & index",
        key="legal_index_page",
        type="primary",
        disabled=not uploads,
    ):
        _, result, error = process_legal_uploads(uploads or [])
        if show_upload_outcome(result, error):
            reload_rag_session()
            st.rerun()

    st.divider()
    st.subheader("Handbook / policy PDFs")
    handbooks = list_handbook_pdfs()
    if handbooks:
        st.caption(f"Found {len(handbooks)} PDF(s) under `{settings.policies_dir}`:")
        for path in handbooks[:10]:
            st.text(f"• {path.name}")
        if len(handbooks) > 10:
            st.caption(f"… and {len(handbooks) - 10} more")
    else:
        st.warning(f"No PDFs in `{settings.policies_dir}` — add handbook files first.")

    if st.button("Run handbook indexing", key="handbook_index_page", type="secondary"):
        with st.spinner("Running scripts/index_documents.py…"):
            ok, output = run_handbook_indexing()
        if ok:
            st.success("Handbook indexing finished.")
            reload_rag_session()
        else:
            st.error("Handbook indexing failed.")
        st.code(output, language="text")