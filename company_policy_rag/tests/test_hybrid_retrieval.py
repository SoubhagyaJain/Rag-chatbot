"""Tests for hybrid RRF fusion."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.hybrid_retrieval import HybridRetriever, reciprocal_rank_fusion


def _nws(node_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=f"text {node_id}", id_=node_id), score=score)


def test_rrf_merges_and_dedupes() -> None:
    dense = [_nws("a", 0.9), _nws("b", 0.8), _nws("c", 0.7)]
    bm25 = [_nws("b", 5.0), _nws("d", 4.0), _nws("a", 3.0)]
    fused = reciprocal_rank_fusion([dense, bm25], rrf_k=60)
    ids = [n.node.node_id for n in fused]
    assert "b" in ids
    assert "a" in ids
    assert "d" in ids
    assert len(ids) == len(set(ids))


def test_hybrid_retriever_dense_only_when_bm25_disabled(monkeypatch) -> None:
    monkeypatch.setattr("src.hybrid_retrieval.settings.enable_hybrid_bm25", False)

    class _Dense:
        def retrieve(self, query):
            return [_nws("only_dense", 1.0)]

    result = HybridRetriever(_Dense()).retrieve(QueryBundle(query_str="test"))
    assert len(result) == 1
    assert result[0].node.node_id == "only_dense"