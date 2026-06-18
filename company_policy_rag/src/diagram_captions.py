"""
Diagram caption nodes for figure/diagram retrieval.

Captions are embedded in Chroma so queries like "agent workflow diagram"
retrieve descriptive text even when the figure itself is not embedded.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document, TextNode

from src.config import settings
from src.utils import logger

_FIGURE_RE = re.compile(
    r"(figure\s+\d+[^\n]{0,120}|diagram[^\n]{0,120}|illustration[^\n]{0,120})",
    re.IGNORECASE,
)


def _manifest_path(file_hash: str) -> Path:
    return settings.pdf_images_dir / file_hash / "manifest.json"


def _load_manifest(file_hash: str) -> dict[str, Any]:
    path = _manifest_path(file_hash)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


_SECTION_META_KEYS = ("section_path", "section_title", "section_number", "section_level")


def _page_section_metadata(
    documents: list[Document],
    page_number: int,
) -> dict[str, Any]:
    """Inherit section metadata from the page document (or nearest page with a path)."""
    for doc in documents:
        meta = doc.metadata or {}
        if meta.get("page_number") == page_number and meta.get("section_path"):
            return {k: meta[k] for k in _SECTION_META_KEYS if meta.get(k) is not None}

    best: dict[str, Any] | None = None
    best_dist = 10**9
    for doc in documents:
        meta = doc.metadata or {}
        pn = meta.get("page_number")
        if not isinstance(pn, int) or not meta.get("section_path"):
            continue
        dist = abs(pn - page_number)
        if dist < best_dist:
            best_dist = dist
            best = {k: meta[k] for k in _SECTION_META_KEYS if meta.get(k) is not None}
    return best or {}


def _surrounding_context(documents: list[Document], page_number: int | None, *, window: int = 3) -> str:
    if page_number is None:
        return ""
    pages = sorted(
        [d for d in documents if d.metadata.get("page_number") == page_number],
        key=lambda d: d.metadata.get("page_number", 0),
    )
    if pages:
        return (pages[0].text or "")[:2000]

    nearby: list[str] = []
    for doc in documents:
        pn = doc.metadata.get("page_number")
        if isinstance(pn, int) and abs(pn - page_number) <= window:
            text = (doc.text or "").strip()
            if text:
                nearby.append(text[:800])
    return "\n".join(nearby)


def _heuristic_caption(context: str, *, source_file: str, page_number: int | None) -> str:
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    figure_lines = [ln for ln in lines if _FIGURE_RE.search(ln)]
    if figure_lines:
        caption = figure_lines[0]
    elif lines:
        caption = lines[0][:200]
    else:
        caption = f"Visual content from {source_file}"

    page_bit = f" (page {page_number})" if page_number else ""
    return f"[DIAGRAM CAPTION — {source_file}{page_bit}] {caption}"


def _llm_caption(context: str, *, source_file: str, page_number: int | None) -> str:
    """Optional Ollama caption when ENABLE_CAPTION_LLM=true."""
    try:
        from llama_index.llms.ollama import Ollama

        llm = Ollama(
            model=settings.caption_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            request_timeout=60.0,
        )
        page_bit = f"page {page_number}" if page_number else "unknown page"
        prompt = (
            f"Write one sentence describing the diagram or figure on {page_bit} "
            f"of {source_file}. Use only the context below; do not invent details.\n\n"
            f"Context:\n{context[:1500]}\n\nCaption:"
        )
        response = llm.complete(prompt)
        text = str(response).strip()
        if text:
            return f"[DIAGRAM CAPTION — {source_file}] {text}"
    except Exception as exc:
        logger.debug("LLM caption failed: %s", exc)
    return _heuristic_caption(context, source_file=source_file, page_number=page_number)


def build_caption_nodes(
    documents: list[Document],
    *,
    file_hash: str,
    source_file: str,
) -> list[TextNode]:
    """Build embeddable caption nodes from PDF image manifest + page context."""
    if not settings.enable_diagram_captions:
        return []

    manifest = _load_manifest(file_hash)
    pages_with_images: set[int] = set()

    pages_data = manifest.get("pages", {})
    if isinstance(pages_data, dict):
        for page_key, page_entry in pages_data.items():
            if not isinstance(page_entry, dict):
                continue
            try:
                pn = int(page_key)
            except (TypeError, ValueError):
                continue
            if page_entry.get("embedded") or page_entry.get("thumbnail"):
                pages_with_images.add(pn)
    elif isinstance(pages_data, list):
        for page_entry in pages_data:
            if not isinstance(page_entry, dict):
                continue
            pn = page_entry.get("page_number")
            if isinstance(pn, int) and (
                page_entry.get("images") or page_entry.get("embedded") or page_entry.get("thumbnail")
            ):
                pages_with_images.add(pn)

    if not pages_with_images:
        return []

    nodes: list[TextNode] = []
    for page_number in sorted(pages_with_images):
        context = _surrounding_context(documents, page_number)
        if settings.enable_caption_llm and context.strip():
            caption_text = _llm_caption(
                context, source_file=source_file, page_number=page_number
            )
        else:
            caption_text = _heuristic_caption(
                context, source_file=source_file, page_number=page_number
            )

        parent_id = f"caption_parent_{file_hash}_{page_number}"
        node_id = f"caption_{uuid.uuid4().hex[:12]}"
        meta = {
            "source_file": source_file,
            "file_hash": file_hash,
            "page_number": page_number,
            "node_role": "caption",
            "parent_id": parent_id,
            "content_type": "diagram_caption",
            "document_type": documents[0].metadata.get("document_type") if documents else "unknown",
        }
        meta.update(_page_section_metadata(documents, page_number))
        nodes.append(TextNode(text=caption_text, metadata=meta, id_=node_id))

    logger.info("Created %d diagram caption nodes for %s", len(nodes), source_file)
    return nodes