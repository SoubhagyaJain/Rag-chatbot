"""
Hybrid dense + BM25 retrieval with reciprocal rank fusion.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.bm25_index import get_bm25_index, sync_bm25_with_chroma
from src.config import settings
from src.timing import get_current_timing, record_stage
from src.utils import logger, timer


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    *,
    rrf_k: int | None = None,
) -> list[NodeWithScore]:
    """
    Merge multiple ranked node lists using RRF.

    score(node) = sum(1 / (k + rank)) across lists; preserves best original score.
    """
    k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
    scores: dict[str, float] = {}
    best_nodes: dict[str, NodeWithScore] = {}

    for node_list in ranked_lists:
        for rank, nws in enumerate(node_list, start=1):
            node_id = nws.node.node_id
            if not node_id:
                continue
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
            existing = best_nodes.get(node_id)
            if existing is None or (nws.score or 0.0) > (existing.score or 0.0):
                best_nodes[node_id] = nws

    if not scores:
        return []

    merged: list[NodeWithScore] = []
    for node_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        nws = best_nodes[node_id]
        merged.append(NodeWithScore(node=nws.node, score=rrf_score))
    return merged


def bm25_nodes_for_query(query: str, top_k: int | None = None) -> list[NodeWithScore]:
    """Run BM25 search and return NodeWithScore list."""
    index = sync_bm25_with_chroma() or get_bm25_index()
    if index is None:
        return []

    k = top_k or settings.bm25_top_k
    hits = index.search(query, top_k=k)
    nodes: list[NodeWithScore] = []
    for node_id, score in hits:
        entry = index.get_entry(node_id)
        if not entry:
            continue
        node = TextNode(text=entry.text, metadata=entry.metadata, id_=entry.node_id)
        nodes.append(NodeWithScore(node=node, score=score))
    return nodes


class HybridRetriever:
    """
    Wraps a dense VectorIndexRetriever with optional BM25 + RRF fusion.
    """

    def __init__(self, dense_retriever: Any) -> None:
        self._dense = dense_retriever

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=query)
        query_str = bundle.query_str

        with timer("chroma_retrieve") as t_dense:
            dense_nodes = self._dense.retrieve(bundle)
        if get_current_timing() is not None:
            record_stage("chroma_retrieve", t_dense["elapsed_ms"])

        if not settings.enable_hybrid_bm25:
            return dense_nodes

        with timer("bm25_retrieve") as t_bm25:
            bm25_nodes = bm25_nodes_for_query(query_str)
        if get_current_timing() is not None:
            record_stage("bm25_retrieve", t_bm25["elapsed_ms"])

        if not bm25_nodes:
            logger.debug("BM25 returned no hits — using dense-only")
            return dense_nodes

        with timer("hybrid_fusion") as t_fusion:
            fused = reciprocal_rank_fusion([dense_nodes, bm25_nodes])
        if get_current_timing() is not None:
            record_stage("hybrid_fusion", t_fusion["elapsed_ms"])

        logger.debug(
            "Hybrid fusion: dense=%d bm25=%d fused=%d",
            len(dense_nodes),
            len(bm25_nodes),
            len(fused),
        )
        return fused

    async def aretrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        if hasattr(self._dense, "aretrieve"):
            bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=query)
            dense_nodes = await self._dense.aretrieve(bundle)
            if not settings.enable_hybrid_bm25:
                return dense_nodes
            bm25_nodes = bm25_nodes_for_query(bundle.query_str)
            if not bm25_nodes:
                return dense_nodes
            return reciprocal_rank_fusion([dense_nodes, bm25_nodes])
        return self.retrieve(query)