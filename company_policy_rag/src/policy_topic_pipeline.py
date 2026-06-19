"""Policy handbook topic pipeline: dress code and other vocabulary-mismatch topics."""

from __future__ import annotations

from enum import Enum

from llama_index.core.schema import NodeWithScore

from src.config import settings
from src.hybrid_retrieval import bm25_nodes_for_query

_DRESS_QUERY_MARKERS = (
    "dress code",
    "dress policy",
    "attire",
    "grooming",
    "appearance",
    "what to wear",
)

_DRESS_PRIMARY_MARKERS = (
    "dress and grooming standards",
    "clean, neat, and professional appearance",
    "dress according to the requirements of their positions",
)

_DRESS_SECONDARY_MARKERS = (
    "professional appearance",
    "dress and grooming",
    "medical or religious accommodation",
)

_DRESS_BM25_QUERY = (
    "Dress and Grooming Standards professional appearance clean neat attire grooming"
)


class PolicyTopicKind(str, Enum):
    NONE = "none"
    DRESS_CODE = "dress_code"


def classify_policy_topic_query(query: str) -> PolicyTopicKind:
    """Classify handbook policy topics that dense retrieval often misses."""
    if not query.strip():
        return PolicyTopicKind.NONE

    q = query.lower()
    if any(marker in q for marker in _DRESS_QUERY_MARKERS):
        return PolicyTopicKind.DRESS_CODE
    return PolicyTopicKind.NONE


def _node_text(nws: NodeWithScore) -> str:
    meta = nws.node.metadata or {}
    section = str(meta.get("section_path") or meta.get("section_title") or "")
    return f"{section} {(nws.node.get_content() or '')}".lower()


def _matches_markers(nws: NodeWithScore, markers: tuple[str, ...]) -> bool:
    text = _node_text(nws)
    return any(marker in text for marker in markers)


def _boost_score_for_pool(nodes: list[NodeWithScore]) -> float:
    max_score = max((n.score or 0.0 for n in nodes), default=1.0)
    return max(max_score * settings.code_boost_score_multiplier, 10.0)


def _boost_nodes_matching_markers(
    nodes: list[NodeWithScore],
    markers: tuple[str, ...],
    boost_score: float,
) -> list[NodeWithScore]:
    boosted: list[NodeWithScore] = []
    for nws in nodes:
        if _matches_markers(nws, markers):
            boosted.append(NodeWithScore(node=nws.node, score=boost_score))
        else:
            boosted.append(nws)
    return boosted


def _inject_bm25_policy_nodes(
    nodes: list[NodeWithScore],
    *,
    bm25_query: str,
    primary_markers: tuple[str, ...],
) -> list[NodeWithScore]:
    boost_score = _boost_score_for_pool(nodes)
    nodes = _boost_nodes_matching_markers(nodes, primary_markers, boost_score)

    if any(_matches_markers(n, primary_markers) for n in nodes):
        return nodes

    if not settings.enable_hybrid_bm25:
        return nodes

    present_ids = {n.node.node_id for n in nodes if n.node.node_id}
    to_inject: list[NodeWithScore] = []
    for hit in bm25_nodes_for_query(bm25_query, top_k=8):
        if not _matches_markers(hit, primary_markers):
            continue
        node_id = hit.node.node_id
        if not node_id or node_id in present_ids:
            continue
        to_inject.append(NodeWithScore(node=hit.node, score=boost_score))
        present_ids.add(node_id)
        break

    return to_inject + nodes if to_inject else nodes


def inject_policy_topic_chunks(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Boost or inject policy topic chunks before reranking."""
    kind = classify_policy_topic_query(query)
    if kind == PolicyTopicKind.DRESS_CODE:
        return _inject_bm25_policy_nodes(
            nodes,
            bm25_query=_DRESS_BM25_QUERY,
            primary_markers=_DRESS_PRIMARY_MARKERS,
        )
    return nodes


def ensure_policy_topic_in_results(
    filtered: list[NodeWithScore],
    query: str,
    candidate_pool: list[NodeWithScore],
) -> list[NodeWithScore]:
    """Re-insert on-topic policy chunks when the cross-encoder drops them."""
    kind = classify_policy_topic_query(query)
    if kind == PolicyTopicKind.NONE:
        return filtered

    primary_markers = (
        _DRESS_PRIMARY_MARKERS
        if kind == PolicyTopicKind.DRESS_CODE
        else ()
    )
    if not primary_markers:
        return filtered

    if any(_matches_markers(n, primary_markers) for n in filtered):
        return filtered

    present = {n.node.node_id for n in filtered if n.node.node_id}
    extras: list[NodeWithScore] = []
    for nws in candidate_pool:
        node_id = nws.node.node_id
        if not node_id or node_id in present:
            continue
        if _matches_markers(nws, primary_markers):
            boost = _boost_score_for_pool(filtered or candidate_pool)
            extras.append(NodeWithScore(node=nws.node, score=boost))
            present.add(node_id)
            break

    if not extras and settings.enable_hybrid_bm25:
        bm25_query = (
            _DRESS_BM25_QUERY
            if kind == PolicyTopicKind.DRESS_CODE
            else ""
        )
        if bm25_query:
            boost = _boost_score_for_pool(filtered or candidate_pool)
            for hit in bm25_nodes_for_query(bm25_query, top_k=8):
                if not _matches_markers(hit, primary_markers):
                    continue
                node_id = hit.node.node_id
                if not node_id or node_id in present:
                    continue
                extras.append(NodeWithScore(node=hit.node, score=max(boost, 10.0)))
                present.add(node_id)
                break

    return extras + filtered if extras else filtered


def reorder_context_for_policy_topic(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Put the defining policy chunk first so Source 1 matches the question."""
    kind = classify_policy_topic_query(query)
    if kind == PolicyTopicKind.NONE or not nodes:
        return nodes

    primary_markers = (
        _DRESS_PRIMARY_MARKERS
        if kind == PolicyTopicKind.DRESS_CODE
        else ()
    )
    if not primary_markers:
        return nodes

    primary = [n for n in nodes if _matches_markers(n, primary_markers)]
    secondary = [
        n
        for n in nodes
        if _matches_markers(n, _DRESS_SECONDARY_MARKERS)
        and n not in primary
    ]
    rest = [n for n in nodes if n not in primary and n not in secondary]
    if primary:
        return primary + secondary + rest
    if secondary:
        return secondary + rest
    return nodes


def finalize_policy_topic_context(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """Post-parent reorder plus BM25 top-up for collapsed policy topic chunks."""
    kind = classify_policy_topic_query(query)
    if kind == PolicyTopicKind.NONE:
        return nodes

    primary_markers = (
        _DRESS_PRIMARY_MARKERS
        if kind == PolicyTopicKind.DRESS_CODE
        else ()
    )
    nodes = reorder_context_for_policy_topic(nodes, query)
    if any(_matches_markers(n, primary_markers) for n in nodes):
        return nodes

    if not settings.enable_hybrid_bm25:
        return nodes

    bm25_query = (
        _DRESS_BM25_QUERY if kind == PolicyTopicKind.DRESS_CODE else ""
    )
    if not bm25_query:
        return nodes

    present = {n.node.node_id for n in nodes if n.node.node_id}
    for hit in bm25_nodes_for_query(bm25_query, top_k=8):
        if not _matches_markers(hit, primary_markers):
            continue
        node_id = hit.node.node_id
        if not node_id or node_id in present:
            continue
        boost = max((n.score or 0.0 for n in nodes), default=1.0)
        return [NodeWithScore(node=hit.node, score=max(boost, 10.0))] + nodes

    return nodes