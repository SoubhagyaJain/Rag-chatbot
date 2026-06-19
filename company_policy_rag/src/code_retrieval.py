"""Inject guidebook code chunks into retrieval candidates for code/tool queries."""

from __future__ import annotations

from functools import lru_cache

from llama_index.core.schema import NodeWithScore, TextNode

from src.bm25_index import get_bm25_index, sync_bm25_with_chroma
from src.config import settings
from src.hybrid_retrieval import bm25_nodes_for_query
from src.query_processing import is_code_or_tool_query

_CURRENCY_QUERY_MARKERS = (
    "currency",
    "convert_currency",
    "conversion tool",
    "currencyconverter",
    "exchange rate",
)

_CURRENCY_BM25_QUERY = (
    "CurrencyConverterTool currency conversion exchange rate real-time currency tool"
)

_CURRENCY_TEXT_MARKERS = (
    "currencyconverter",
    "currency conversion",
    "exchange rate",
    "currency_analyst",
)

_CODE_LINK_MARKERS = (
    "code is available",
    "check this code",
    "dailydoseofds.com/p/",
)

_CHECK_THIS_OUT_MARKERS = (
    "check this out",
)

_CUSTOM_TOOL_MARKERS = (
    "custom tools via mcp",
    "build custom tools",
    "custom tools at times",
    "@mcp.tool",
    "mcp.tool()",
)


@lru_cache(maxsize=1)
def _cached_guidebook_code_node_ids() -> tuple[str, ...]:
    """All indexed node IDs with content_type=code (guidebook has ~3)."""
    index = sync_bm25_with_chroma() or get_bm25_index()
    if index is None:
        return ()
    ids: list[str] = []
    for entry in index.entries:
        if entry.metadata.get("content_type") == "code":
            ids.append(entry.node_id)
    return tuple(ids)


def get_guidebook_code_nodes() -> list[NodeWithScore]:
    """Return all content_type=code chunks from the BM25 corpus."""
    index = sync_bm25_with_chroma() or get_bm25_index()
    if index is None:
        return []

    nodes: list[NodeWithScore] = []
    for node_id in _cached_guidebook_code_node_ids():
        entry = index.get_entry(node_id)
        if not entry:
            continue
        node = TextNode(text=entry.text, metadata=entry.metadata, id_=entry.node_id)
        nodes.append(NodeWithScore(node=node, score=0.0))
    return nodes


def clear_code_node_cache() -> None:
    """Clear cached code node IDs (for tests after index rebuild)."""
    _cached_guidebook_code_node_ids.cache_clear()


def inject_code_chunks(
    nodes: list[NodeWithScore],
    query: str,
    *,
    min_code: int | None = None,
) -> list[NodeWithScore]:
    """
    Prepend missing code chunks when the query signals code/tool intent.

    Ensures at least min_code code nodes reach the reranker without dropping
    existing candidates.
    """
    if not settings.enable_code_retrieval_boost or not is_code_or_tool_query(query):
        return nodes

    target_min = min_code if min_code is not None else settings.code_chunk_inject_min
    if target_min <= 0:
        return nodes

    present_ids = {n.node.node_id for n in nodes if n.node.node_id}
    code_nodes = get_guidebook_code_nodes()
    if not code_nodes:
        return nodes

    code_in_pool = sum(
        1
        for n in nodes
        if (n.node.metadata or {}).get("content_type") == "code"
    )
    if code_in_pool >= target_min:
        return nodes

    boost_score = _boost_score_for_pool(nodes)

    to_inject: list[NodeWithScore] = []
    for code_nws in code_nodes:
        node_id = code_nws.node.node_id
        if not node_id or node_id in present_ids:
            continue
        to_inject.append(NodeWithScore(node=code_nws.node, score=boost_score))
        if code_in_pool + len(to_inject) >= target_min:
            break

    if not to_inject:
        return nodes

    return to_inject + nodes


def count_code_nodes(nodes: list[NodeWithScore]) -> int:
    return sum(
        1 for n in nodes if (n.node.metadata or {}).get("content_type") == "code"
    )


def _is_currency_query(query: str) -> bool:
    q = query.lower()
    return any(marker in q for marker in _CURRENCY_QUERY_MARKERS)


def _boost_score_for_pool(nodes: list[NodeWithScore]) -> float:
    max_score = max((n.score or 0.0 for n in nodes), default=1.0)
    return max(max_score * settings.code_boost_score_multiplier, 10.0)


def _boost_nodes_matching_markers(
    nodes: list[NodeWithScore],
    text_markers: tuple[str, ...],
    boost_score: float,
) -> list[NodeWithScore]:
    boosted: list[NodeWithScore] = []
    for nws in nodes:
        text = (nws.node.get_content() or "").lower()
        if any(marker in text for marker in text_markers):
            boosted.append(NodeWithScore(node=nws.node, score=boost_score))
        else:
            boosted.append(nws)
    return boosted


def _inject_bm25_topic_nodes(
    nodes: list[NodeWithScore],
    *,
    bm25_query: str,
    min_inject: int,
    text_markers: tuple[str, ...],
) -> list[NodeWithScore]:
    if min_inject <= 0:
        return nodes

    boost_score = _boost_score_for_pool(nodes)
    nodes = _boost_nodes_matching_markers(nodes, text_markers, boost_score)

    present_ids = {n.node.node_id for n in nodes if n.node.node_id}
    marker_hits = sum(
        1
        for n in nodes
        if any(marker in (n.node.get_content() or "").lower() for marker in text_markers)
    )
    if marker_hits >= min_inject:
        return nodes

    hits = bm25_nodes_for_query(bm25_query, top_k=8)
    to_inject: list[NodeWithScore] = []
    for hit in hits:
        text = (hit.node.get_content() or "").lower()
        if not any(marker in text for marker in text_markers):
            continue
        node_id = hit.node.node_id
        if not node_id or node_id in present_ids:
            continue
        to_inject.append(NodeWithScore(node=hit.node, score=boost_score))
        present_ids.add(node_id)
        if marker_hits + len(to_inject) >= min_inject:
            break

    return to_inject + nodes if to_inject else nodes


def inject_retrieval_boost_chunks(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Inject code chunks and currency-topic prose for code/tool queries."""
    if not settings.enable_code_retrieval_boost or not is_code_or_tool_query(query):
        return nodes

    boosted = inject_code_chunks(nodes, query)
    if _is_currency_query(query):
        boosted = _inject_bm25_topic_nodes(
            boosted,
            bm25_query=_CURRENCY_BM25_QUERY,
            min_inject=settings.code_chunk_inject_min,
            text_markers=_CURRENCY_TEXT_MARKERS,
        )
    q = query.lower()
    if "custom tool" in q or "build custom" in q:
        boosted = _inject_bm25_topic_nodes(
            boosted,
            bm25_query="custom tools via MCP @mcp.tool function implementation",
            min_inject=1,
            text_markers=_CUSTOM_TOOL_MARKERS,
        )
    return boosted


def _promote_nodes_with_markers(
    nodes: list[NodeWithScore],
    markers: tuple[str, ...],
) -> list[NodeWithScore]:
    if not nodes or not markers:
        return nodes
    primary: list[NodeWithScore] = []
    rest: list[NodeWithScore] = []
    for nws in nodes:
        text = (nws.node.get_content() or "").lower()
        if any(marker in text for marker in markers):
            primary.append(nws)
        else:
            rest.append(nws)
    return primary + rest if primary else nodes


def ensure_topic_nodes_in_results(
    filtered: list[NodeWithScore],
    query: str,
    candidate_pool: list[NodeWithScore],
) -> list[NodeWithScore]:
    """Re-insert on-topic chunks dropped by the cross-encoder reranker."""
    if not settings.enable_code_retrieval_boost or not is_code_or_tool_query(query):
        return filtered

    marker_groups: list[tuple[str, ...]] = []
    if _is_currency_query(query):
        marker_groups.append(_CURRENCY_TEXT_MARKERS)
    q = query.lower()
    if "code is available" in q or "full code" in q or ("point" in q and "code" in q):
        marker_groups.append(_CODE_LINK_MARKERS)
    if "check this out" in q or "walkthrough" in q:
        marker_groups.append(_CHECK_THIS_OUT_MARKERS)
    if "custom tool" in q or "build custom" in q:
        marker_groups.append(_CUSTOM_TOOL_MARKERS)

    if not marker_groups:
        return filtered

    present = {n.node.node_id for n in filtered if n.node.node_id}
    extras: list[NodeWithScore] = []
    for markers in marker_groups:
        for nws in candidate_pool:
            text = (nws.node.get_content() or "").lower()
            node_id = nws.node.node_id
            if not node_id or node_id in present:
                continue
            if any(marker in text for marker in markers):
                extras.append(nws)
                present.add(node_id)

    if not extras:
        return filtered
    return extras + filtered


def promote_code_tool_nodes(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Move on-topic code/currency chunks to the front of the final context."""
    from src.tool_code_pipeline import reorder_context_for_tool_code

    return reorder_context_for_tool_code(nodes, query)