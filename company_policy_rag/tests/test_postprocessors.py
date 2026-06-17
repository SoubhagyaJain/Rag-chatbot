"""Tests for retrieval post-processors."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.postprocessors import RelativeScoreThresholdPostprocessor


def _node(score: float, text: str = "chunk") -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text), score=score)


def test_relative_score_filter_drops_weak_chunks() -> None:
    processor = RelativeScoreThresholdPostprocessor(min_ratio=0.5, min_keep=1)
    nodes = [
        _node(10.0, "best"),
        _node(6.0, "good"),
        _node(2.0, "weak"),
        _node(1.0, "noise"),
    ]
    result = processor.postprocess_nodes(nodes)
    texts = [n.get_content() for n in result]
    assert "best" in texts
    assert "good" in texts
    assert "noise" not in texts


def test_relative_score_keeps_min_one() -> None:
    processor = RelativeScoreThresholdPostprocessor(min_ratio=0.9, min_keep=1)
    nodes = [_node(5.0, "only"), _node(1.0, "drop")]
    result = processor.postprocess_nodes(nodes)
    assert len(result) >= 1