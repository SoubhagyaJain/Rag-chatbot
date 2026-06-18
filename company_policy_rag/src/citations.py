"""
Citation selection for policy RAG — precision over recall.

Production-rag principle: show only sources that grounded the answer.
- Generation nodes come from the query engine (post-rerank/filter), not a second retrieval.
- When the answer cites [Source N], only those indices are displayed.
- Otherwise fall back to a strict relevance-score threshold on generation nodes.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any, Sequence

from llama_index.core.schema import NodeWithScore

from src.config import settings
from src.query_processing import is_comprehensive_list_query
from src.utils import format_citations, logger, prepare_citations_for_display

# Nodes passed to the LLM during policy_search in the current agent turn.
_generation_nodes_this_turn: ContextVar[list[NodeWithScore] | None] = ContextVar(
    "generation_nodes_this_turn", default=None
)

_SOURCE_TAG_PATTERN = re.compile(r"\[Source\s+([^\]]+)\]", re.IGNORECASE)


def begin_citation_turn() -> None:
    """Reset per-turn source tracking before an agent or query-engine call."""
    _generation_nodes_this_turn.set([])


def record_generation_sources(nodes: Sequence[NodeWithScore]) -> None:
    """
    Append nodes from a query-engine response to the current turn.

    Called by SourceTrackingQueryEngine after each policy_search — these are the
    exact chunks that were reranked, filtered, and passed to synthesis.
    """
    if not nodes:
        return
    current = _generation_nodes_this_turn.get()
    if current is None:
        current = []
    seen = {id(n.node) for n in current}
    for node in nodes:
        node_id = id(node.node)
        if node_id not in seen:
            current.append(node)
            seen.add(node_id)
    _generation_nodes_this_turn.set(current)


def get_generation_nodes_this_turn() -> list[NodeWithScore]:
    """Return accumulated generation nodes for the active turn."""
    return list(_generation_nodes_this_turn.get() or [])


def extract_cited_source_indices(answer: str) -> set[int]:
    """
    Parse 1-based [Source N] tags from a grounded answer.

    Handles: [Source 1], [Source 2, Source 3], [Source 1, excerpt 1]
    """
    indices: set[int] = set()
    for match in _SOURCE_TAG_PATTERN.finditer(answer):
        inner = match.group(1)
        for num_match in re.finditer(r"\b(\d+)\b", inner):
            indices.add(int(num_match.group(1)))
    return indices


def _node_label(node: NodeWithScore, index: int | None = None) -> str:
    meta = node.metadata or getattr(node.node, "metadata", None) or {}
    section = meta.get("section_path") or meta.get("section_title") or "?"
    page = meta.get("page_number")
    prefix = f"#{index} " if index is not None else ""
    page_str = f" p.{page}" if page is not None else ""
    score = f" score={node.score:.3f}" if node.score is not None else ""
    return f"{prefix}{section}{page_str}{score}"


def log_retrieval_stage(stage: str, nodes: Sequence[NodeWithScore]) -> None:
    """Structured log of chunk counts and top sections at each pipeline stage."""
    if not settings.enable_citation_pipeline_logging:
        return
    if not nodes:
        logger.info("Citation pipeline | %s | 0 chunks", stage)
        return
    previews = [_node_label(n, i + 1) for i, n in enumerate(nodes[:8])]
    suffix = " …" if len(nodes) > 8 else ""
    logger.info(
        "Citation pipeline | %s | %d chunks | %s%s",
        stage,
        len(nodes),
        " | ".join(previews),
        suffix,
    )


def filter_nodes_by_relevance_score(
    nodes: list[NodeWithScore],
    min_ratio: float,
) -> list[NodeWithScore]:
    """Keep nodes scoring at least min_ratio × top reranker score."""
    scored = [n for n in nodes if n.score is not None]
    if not scored:
        return nodes[:1] if nodes else []

    top_score = max(n.score for n in scored if n.score is not None)
    if top_score <= 0:
        return sorted(scored, key=lambda n: n.score or 0, reverse=True)[:1]

    threshold = top_score * min_ratio
    filtered = [n for n in nodes if n.score is not None and n.score >= threshold]
    if not filtered:
        return sorted(scored, key=lambda n: n.score or 0, reverse=True)[:1]
    return filtered


def select_citations_for_answer(
    answer: str,
    generation_nodes: list[NodeWithScore],
    *,
    user_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Select UI citations from generation nodes only (never a parallel retrieval).

    Priority:
    1. Explicit [Source N] tags in the answer → only those indices (1-based).
    2. No tags → top nodes above CITATION_MIN_RELEVANCE_RATIO (strict fallback).
    """
    if not generation_nodes:
        logger.info("Citation selection | no generation nodes — returning empty citations")
        return []

    cited_indices = extract_cited_source_indices(answer)
    selected_nodes: list[NodeWithScore] = []
    selection_mode: str

    if cited_indices:
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(generation_nodes):
                selected_nodes.append(generation_nodes[idx - 1])
            else:
                logger.warning(
                    "Answer cites [Source %d] but only %d generation chunks available",
                    idx,
                    len(generation_nodes),
                )
        selection_mode = "cited_in_answer"
        if not selected_nodes:
            logger.warning(
                "Citation tags %s did not map to generation nodes — using score fallback",
                sorted(cited_indices),
            )

    if not selected_nodes:
        selected_nodes = filter_nodes_by_relevance_score(
            generation_nodes,
            settings.citation_min_relevance_ratio,
        )
        # Broader fallback for comprehensive list questions; tighter otherwise.
        if user_query and is_comprehensive_list_query(user_query):
            max_fallback = settings.citation_max_sources
        else:
            max_fallback = max(1, min(3, settings.citation_max_sources // 2))
        selected_nodes = sorted(
            selected_nodes,
            key=lambda n: n.score or 0,
            reverse=True,
        )[:max_fallback]
        selection_mode = "score_threshold_fallback"

    citations = format_citations(selected_nodes)
    for citation in citations:
        citation["selection_reason"] = selection_mode

    if settings.enable_citation_pipeline_logging:
        log_retrieval_stage("generation_input", generation_nodes)
        log_retrieval_stage("citations_displayed", selected_nodes)
        logger.info(
            "Citation selection | mode=%s | generation=%d | displayed=%d | cited_tags=%s",
            selection_mode,
            len(generation_nodes),
            len(citations),
            sorted(cited_indices) if cited_indices else [],
        )

    return prepare_citations_for_display(citations)