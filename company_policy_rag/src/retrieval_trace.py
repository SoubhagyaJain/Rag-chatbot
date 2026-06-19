"""Build retrieval observability payload for API responses."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore

from src.timing import PipelineTiming, get_current_timing


def _chunk_preview(node: NodeWithScore, index: int) -> dict[str, Any]:
    meta = node.node.metadata or getattr(node.node, "metadata", None) or {}
    text = (node.node.get_content() or "").strip()
    preview = text[:120] + ("…" if len(text) > 120 else "")
    return {
        "index": index,
        "section_path": meta.get("section_path") or meta.get("section_title"),
        "page_number": meta.get("page_number"),
        "source_file": meta.get("source_file"),
        "score": round(float(node.score or 0.0), 4),
        "excerpt_preview": preview,
    }


def build_retrieval_trace(
    nodes: list[NodeWithScore],
    timing: PipelineTiming | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize generation chunks and pipeline stage timings."""
    t = timing
    if t is None:
        current = get_current_timing()
        t = current.as_dict() if current else {}

    stages: dict[str, float] = {}
    if isinstance(t, PipelineTiming):
        stages = t.as_dict()
    elif isinstance(t, dict):
        stages = {k: float(v) for k, v in t.items() if isinstance(v, (int, float))}

    return {
        "chunk_count": len(nodes),
        "chunks": [_chunk_preview(n, i + 1) for i, n in enumerate(nodes[:12])],
        "stages": stages,
    }