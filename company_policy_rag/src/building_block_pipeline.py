"""Lightweight guidebook topic pipeline: classification and context reordering (round 1)."""

from __future__ import annotations

from enum import Enum

from llama_index.core.schema import NodeWithScore

from src.config import settings
from src.hybrid_retrieval import bm25_nodes_for_query

_GUARDRAILS_MARKERS = (
    "guardrails",
    "guardrail",
    "examples of useful guardrails",
    "validation checkpoints",
    "limiting tool usage",
)

_GUARDRAILS_PRIMARY_MARKERS = (
    "5) guardrails",
    "4. guardrails",
    "examples of useful guardrails",
    "without constraints",
    "limiting tool usage",
    "validation checkpoints",
    "agents are powerful but without constraints",
)

_PLANNING_BLOCK_MARKERS = (
    "planning building block",
    "building block",
    "five levels of agentic",
    "six building blocks",
    "subdividing tasks",
    "outlining objectives",
    "#4) planning pattern",
    "planning pattern",
)

# Design-pattern planning — deprioritize when query asks about Planning building block
_PLANNING_PATTERN_MARKERS = (
    "plan-and-execute",
    "plan and execute",
    "design patterns",
    "react",
    "reflection pattern",
    "agentic ai design patterns",
)


class GuidebookTopicKind(str, Enum):
    NONE = "none"
    GUARDRAILS = "guardrails"
    PLANNING_BLOCK = "planning_block"


def classify_guidebook_topic_query(query: str) -> GuidebookTopicKind:
    """Classify round-1 priority guidebook topics (guardrails, planning building block)."""
    if not query.strip():
        return GuidebookTopicKind.NONE

    q = query.lower()

    if "planning" in q and "building block" in q:
        return GuidebookTopicKind.PLANNING_BLOCK

    if "guardrail" in q:
        return GuidebookTopicKind.GUARDRAILS

    return GuidebookTopicKind.NONE


def _node_text(nws: NodeWithScore) -> str:
    meta = nws.node.metadata or {}
    section = str(meta.get("section_path") or meta.get("section_title") or "")
    return f"{section} {(nws.node.get_content() or '')}".lower()


def _matches_markers(nws: NodeWithScore, markers: tuple[str, ...]) -> bool:
    text = _node_text(nws)
    return any(marker in text for marker in markers)


def _promote_matching(
    nodes: list[NodeWithScore],
    markers: tuple[str, ...],
) -> list[NodeWithScore]:
    primary = [n for n in nodes if _matches_markers(n, markers)]
    rest = [n for n in nodes if n not in primary]
    return primary + rest if primary else nodes


_PLANNING_DEFINITION_MARKERS = (
    "subdividing tasks",
    "outlining objectives",
)


def _reorder_planning_block(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Prefer planning definitions (subdividing tasks) over generic pattern mentions."""
    definition_hits = [
        n for n in nodes if _matches_markers(n, _PLANNING_DEFINITION_MARKERS)
    ]
    block_hits = [
        n
        for n in nodes
        if _matches_markers(n, _PLANNING_BLOCK_MARKERS) and n not in definition_hits
    ]
    pattern_hits = [
        n
        for n in nodes
        if _matches_markers(n, _PLANNING_PATTERN_MARKERS)
        and n not in definition_hits
        and n not in block_hits
    ]
    rest = [
        n
        for n in nodes
        if n not in definition_hits and n not in block_hits and n not in pattern_hits
    ]
    if definition_hits:
        return definition_hits + block_hits + rest + pattern_hits
    if block_hits:
        return block_hits + rest + pattern_hits
    return nodes


_BM25_BY_KIND: dict[GuidebookTopicKind, str] = {
    GuidebookTopicKind.GUARDRAILS: (
        "Guardrails building block safety constraints limits validation checkpoints"
    ),
    GuidebookTopicKind.PLANNING_BLOCK: (
        "Planning building block subdividing tasks outlining objectives six building blocks"
    ),
}


def ensure_guidebook_topic_in_results(
    filtered: list[NodeWithScore],
    query: str,
    candidate_pool: list[NodeWithScore],
) -> list[NodeWithScore]:
    """Re-insert on-topic chunks when reranker drops guardrails/planning block prose."""
    kind = classify_guidebook_topic_query(query)
    if kind == GuidebookTopicKind.NONE:
        return filtered

    markers = (
        _GUARDRAILS_MARKERS
        if kind == GuidebookTopicKind.GUARDRAILS
        else _PLANNING_BLOCK_MARKERS
    )
    primary_markers = (
        _GUARDRAILS_PRIMARY_MARKERS
        if kind == GuidebookTopicKind.GUARDRAILS
        else _PLANNING_BLOCK_MARKERS
    )
    if any(_matches_markers(n, primary_markers) for n in filtered):
        return filtered

    present = {n.node.node_id for n in filtered if n.node.node_id}
    extras: list[NodeWithScore] = []
    for nws in candidate_pool:
        node_id = nws.node.node_id
        if not node_id or node_id in present:
            continue
        if _matches_markers(nws, primary_markers):
            extras.append(nws)
            present.add(node_id)
            break

    if not extras and settings.enable_hybrid_bm25:
        bm25_query = _BM25_BY_KIND.get(kind, "")
        if bm25_query:
            boost = max((n.score or 0.0 for n in filtered), default=1.0) * 1.2
            for hit in bm25_nodes_for_query(bm25_query, top_k=6):
                if not _matches_markers(hit, primary_markers):
                    continue
                node_id = hit.node.node_id
                if not node_id or node_id in present:
                    continue
                extras.append(NodeWithScore(node=hit.node, score=max(boost, 10.0)))
                present.add(node_id)
                break

    return extras + filtered if extras else filtered


def reorder_context_for_guidebook_topic(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Reorder context so Source 1 aligns with guardrails or Planning building block topics."""
    if not nodes:
        return nodes

    kind = classify_guidebook_topic_query(query)
    if kind == GuidebookTopicKind.GUARDRAILS:
        primary = [n for n in nodes if _matches_markers(n, _GUARDRAILS_PRIMARY_MARKERS)]
        weak = [
            n
            for n in nodes
            if _matches_markers(n, _GUARDRAILS_MARKERS) and n not in primary
        ]
        rest = [n for n in nodes if n not in primary and n not in weak]
        if primary:
            return primary + weak + rest
        return _promote_matching(nodes, _GUARDRAILS_MARKERS)
    if kind == GuidebookTopicKind.PLANNING_BLOCK:
        return _reorder_planning_block(nodes)
    return nodes


def finalize_guidebook_topic_context(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Post-parent reorder plus BM25 top-up when expand collapsed topic chunks."""
    kind = classify_guidebook_topic_query(query)
    if kind == GuidebookTopicKind.NONE:
        return nodes

    markers = (
        _GUARDRAILS_MARKERS
        if kind == GuidebookTopicKind.GUARDRAILS
        else _PLANNING_BLOCK_MARKERS
    )
    nodes = reorder_context_for_guidebook_topic(nodes, query)
    primary_markers = (
        _GUARDRAILS_PRIMARY_MARKERS
        if kind == GuidebookTopicKind.GUARDRAILS
        else _PLANNING_BLOCK_MARKERS
    )
    if any(_matches_markers(n, primary_markers) for n in nodes):
        return nodes

    if not settings.enable_hybrid_bm25:
        return nodes

    bm25_query = _BM25_BY_KIND.get(kind, "")
    if not bm25_query:
        return nodes

    present = {n.node.node_id for n in nodes if n.node.node_id}
    for hit in bm25_nodes_for_query(bm25_query, top_k=6):
        if not _matches_markers(hit, primary_markers):
            continue
        node_id = hit.node.node_id
        if not node_id or node_id in present:
            continue
        boost = max((n.score or 0.0 for n in nodes), default=1.0)
        return [NodeWithScore(node=hit.node, score=max(boost, 10.0))] + nodes

    return nodes