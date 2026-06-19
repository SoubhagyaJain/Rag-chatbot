"""Guidebook agent-topic pipeline: manager agent, RAG-in-agent, memory building block."""

from __future__ import annotations

from enum import Enum

from llama_index.core.schema import NodeWithScore

from src.config import settings
from src.hybrid_retrieval import bm25_nodes_for_query

_MANAGER_PRIMARY_MARKERS = (
    "manager agent",
    "coordinates multiple sub-agents",
)

_MANAGER_WEAK_MARKERS = (
    "multi-agent pattern",
)

_RAG_WORKFLOW_MARKERS = (
    "agentic rag",
    "retriever agent",
    "vector db",
    "dynamically fetch context",
    "retrieves the relevant context",
)

_MEMORY_BLOCK_MARKERS = (
    "without memory",
    "short-term memory",
    "long-term memory",
    "entity memory",
    "different types of memory",
    "building blocks of ai agents",
)

# Deprioritize unrelated multi-agent project walkthroughs
_MULTI_AGENT_NOISE_MARKERS = (
    "multi-agent hotel finder",
    "multi-agent content creation",
    "multi-agent deep researcher",
    "kickoff and results",
)


class AgentTopicKind(str, Enum):
    NONE = "none"
    MANAGER_AGENT = "manager_agent"
    RAG_IN_AGENT = "rag_in_agent"
    MEMORY_BLOCK = "memory_block"


def classify_agent_topic_query(query: str) -> AgentTopicKind:
    """Classify round-2 agent workflow / memory topic queries."""
    if not query.strip():
        return AgentTopicKind.NONE

    q = query.lower()

    if "manager agent" in q or (
        "multi-agent" in q and "what does" in q and "do" in q
    ):
        return AgentTopicKind.MANAGER_AGENT

    if "rag" in q and "agent" in q:
        return AgentTopicKind.RAG_IN_AGENT

    if "memory" in q and ("building block" in q or "how does memory" in q):
        return AgentTopicKind.MEMORY_BLOCK

    return AgentTopicKind.NONE


def _node_text(nws: NodeWithScore) -> str:
    meta = nws.node.metadata or {}
    section = str(meta.get("section_path") or meta.get("section_title") or "")
    return f"{section} {(nws.node.get_content() or '')}".lower()


def _content_text(nws: NodeWithScore, *, head_chars: int | None = None) -> str:
    text = (nws.node.get_content() or "").lower()
    if head_chars is not None:
        return text[:head_chars]
    return text


def _matches_markers(nws: NodeWithScore, markers: tuple[str, ...]) -> bool:
    text = _node_text(nws)
    return any(marker in text for marker in markers)


def _matches_content_markers(
    nws: NodeWithScore,
    markers: tuple[str, ...],
    *,
    head_chars: int | None = None,
) -> bool:
    text = _content_text(nws, head_chars=head_chars)
    return any(marker in text for marker in markers)


def _has_usable_manager_definition(nws: NodeWithScore) -> bool:
    """True when manager-agent prose is visible (not buried in a parent mega-chunk)."""
    return _matches_content_markers(
        nws,
        _MANAGER_PRIMARY_MARKERS,
        head_chars=2000,
    )


_BM25_BY_KIND: dict[AgentTopicKind, str] = {
    AgentTopicKind.MANAGER_AGENT: (
        "manager agent coordinates sub-agents multi-agent pattern iteratively"
    ),
    AgentTopicKind.RAG_IN_AGENT: (
        "Agentic RAG retriever agent workflow vector DB Firecrawl context"
    ),
    AgentTopicKind.MEMORY_BLOCK: (
        "Memory building block short-term long-term entity memory agents improve"
    ),
}

_PRIMARY_BY_KIND: dict[AgentTopicKind, tuple[str, ...]] = {
    AgentTopicKind.MANAGER_AGENT: _MANAGER_PRIMARY_MARKERS,
    AgentTopicKind.RAG_IN_AGENT: _RAG_WORKFLOW_MARKERS,
    AgentTopicKind.MEMORY_BLOCK: _MEMORY_BLOCK_MARKERS,
}


def _reorder_manager_agent(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    primary = [n for n in nodes if _has_usable_manager_definition(n)]
    weak = [
        n
        for n in nodes
        if _matches_markers(n, _MANAGER_WEAK_MARKERS) and n not in primary
    ]
    noise = [
        n
        for n in nodes
        if _matches_markers(n, _MULTI_AGENT_NOISE_MARKERS)
        and n not in primary
        and n not in weak
    ]
    rest = [n for n in nodes if n not in primary and n not in weak and n not in noise]
    if primary:
        return primary + weak + rest + noise
    if weak:
        return weak + rest + noise
    return nodes


def _reorder_rag_workflow(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    primary = [n for n in nodes if _matches_markers(n, _RAG_WORKFLOW_MARKERS)]
    rest = [n for n in nodes if n not in primary]
    if primary:
        return primary + rest
    return nodes


def _reorder_memory_block(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    detail = [
        n
        for n in nodes
        if _matches_markers(n, ("short-term memory", "long-term memory", "entity memory"))
    ]
    overview = [
        n
        for n in nodes
        if _matches_markers(n, ("without memory", "building blocks of ai agents"))
        and n not in detail
    ]
    rest = [n for n in nodes if n not in detail and n not in overview]
    if detail:
        return detail + overview + rest
    if overview:
        return overview + rest
    return nodes


def reorder_context_for_agent_topic(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Reorder context so Source 1 aligns with manager/RAG/memory topics."""
    if not nodes:
        return nodes

    kind = classify_agent_topic_query(query)
    if kind == AgentTopicKind.MANAGER_AGENT:
        return _reorder_manager_agent(nodes)
    if kind == AgentTopicKind.RAG_IN_AGENT:
        return _reorder_rag_workflow(nodes)
    if kind == AgentTopicKind.MEMORY_BLOCK:
        return _reorder_memory_block(nodes)
    return nodes


def _inject_from_pool(
    nodes: list[NodeWithScore],
    pool: list[NodeWithScore],
    primary_markers: tuple[str, ...],
) -> list[NodeWithScore]:
    if any(_matches_markers(n, primary_markers) for n in nodes):
        return nodes

    present = {n.node.node_id for n in nodes if n.node.node_id}
    for nws in pool:
        node_id = nws.node.node_id
        if not node_id or node_id in present:
            continue
        if _matches_markers(nws, primary_markers):
            return [nws] + nodes

    return nodes


def ensure_agent_topic_in_results(
    filtered: list[NodeWithScore],
    query: str,
    candidate_pool: list[NodeWithScore],
) -> list[NodeWithScore]:
    """Re-insert on-topic chunks when reranker drops manager/RAG/memory prose."""
    kind = classify_agent_topic_query(query)
    if kind == AgentTopicKind.NONE:
        return filtered

    primary_markers = _PRIMARY_BY_KIND[kind]
    if kind == AgentTopicKind.MANAGER_AGENT:
        if any(_has_usable_manager_definition(n) for n in filtered):
            return reorder_context_for_agent_topic(filtered, query)
    elif any(_matches_markers(n, primary_markers) for n in filtered):
        return filtered

    filtered = _inject_from_pool(filtered, candidate_pool, primary_markers)

    if kind == AgentTopicKind.MANAGER_AGENT:
        if any(_has_usable_manager_definition(n) for n in filtered):
            return reorder_context_for_agent_topic(filtered, query)
    elif any(_matches_markers(n, primary_markers) for n in filtered):
        return filtered

    if not settings.enable_hybrid_bm25:
        return filtered

    bm25_query = _BM25_BY_KIND.get(kind, "")
    if not bm25_query:
        return filtered

    present = {n.node.node_id for n in filtered if n.node.node_id}
    boost = max((n.score or 0.0 for n in filtered), default=1.0) * 1.2
    for hit in bm25_nodes_for_query(bm25_query, top_k=8):
        if kind == AgentTopicKind.MANAGER_AGENT:
            if not _has_usable_manager_definition(hit):
                continue
        elif not _matches_markers(hit, primary_markers):
            continue
        node_id = hit.node.node_id
        if not node_id or node_id in present:
            continue
        return reorder_context_for_agent_topic(
            [NodeWithScore(node=hit.node, score=max(boost, 10.0))] + filtered,
            query,
        )

    return filtered


def finalize_agent_topic_context(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Post-parent reorder plus BM25 top-up for agent workflow topics."""
    kind = classify_agent_topic_query(query)
    if kind == AgentTopicKind.NONE:
        return nodes

    primary_markers = _PRIMARY_BY_KIND[kind]
    nodes = reorder_context_for_agent_topic(nodes, query)

    if kind == AgentTopicKind.MANAGER_AGENT:
        if any(_has_usable_manager_definition(n) for n in nodes):
            return nodes
    elif any(_matches_markers(n, primary_markers) for n in nodes):
        return nodes

    if not settings.enable_hybrid_bm25:
        return nodes

    bm25_query = _BM25_BY_KIND.get(kind, "")
    if not bm25_query:
        return nodes

    present = {n.node.node_id for n in nodes if n.node.node_id}
    boost = max((n.score or 0.0 for n in nodes), default=1.0)
    for hit in bm25_nodes_for_query(bm25_query, top_k=8):
        if kind == AgentTopicKind.MANAGER_AGENT:
            if not _has_usable_manager_definition(hit):
                continue
        elif not _matches_markers(hit, primary_markers):
            continue
        node_id = hit.node.node_id
        if not node_id or node_id in present:
            continue
        return reorder_context_for_agent_topic(
            [NodeWithScore(node=hit.node, score=max(boost, 10.0))] + nodes,
            query,
        )

    return nodes