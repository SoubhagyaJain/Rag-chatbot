"""Tests for policy handbook topic classification and retrieval boost."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.policy_topic_pipeline import (
    PolicyTopicKind,
    classify_policy_topic_query,
    ensure_policy_topic_in_results,
    inject_policy_topic_chunks,
    reorder_context_for_policy_topic,
)

_DRESS_TEXT = (
    "Dress and Grooming Standards contribute to morale. During business hours, "
    "employees are expected to present a clean, neat, and professional appearance "
    "and to dress according to the requirements of their positions."
)


def _node(
    text: str,
    *,
    section: str = "",
    node_id: str = "n1",
    page: int | None = None,
) -> NodeWithScore:
    meta: dict[str, object] = {}
    if section:
        meta["section_path"] = section
    if page is not None:
        meta["page_number"] = page
    return NodeWithScore(
        node=TextNode(text=text, metadata=meta, id_=node_id),
        score=0.03,
    )


def test_classify_dress_code_query() -> None:
    kind = classify_policy_topic_query("What is the dress code policy?")
    assert kind == PolicyTopicKind.DRESS_CODE


def test_inject_boosts_dress_chunk_score() -> None:
    dress = _node(
        _DRESS_TEXT,
        section="E. Dress and Grooming Standards",
        node_id="dress",
        page=31,
    )
    ethics = _node(
        "Employees must act ethically and follow conduct rules.",
        section="J. Ethics and Conduct",
        node_id="ethics",
        page=34,
    )
    result = inject_policy_topic_chunks([ethics, dress], "What is the dress code policy?")
    dress_result = next(n for n in result if n.node.node_id == "dress")
    ethics_result = next(n for n in result if n.node.node_id == "ethics")
    assert dress_result.score is not None
    assert ethics_result.score is not None
    assert dress_result.score > ethics_result.score


def test_ensure_reinserts_dress_chunk_dropped_by_reranker() -> None:
    dress = _node(
        _DRESS_TEXT,
        section="E. Dress and Grooming Standards",
        node_id="dress",
        page=31,
    )
    ethics = _node(
        "Employees must act ethically.",
        section="J. Ethics and Conduct",
        node_id="ethics",
        page=34,
    )
    pool = [ethics, dress]
    filtered = [ethics]
    result = ensure_policy_topic_in_results(
        filtered,
        "What is the dress code policy?",
        pool,
    )
    assert result[0].node.node_id == "dress"


def test_reorder_puts_dress_policy_first() -> None:
    dress = _node(
        _DRESS_TEXT,
        section="E. Dress and Grooming Standards",
        node_id="dress",
        page=31,
    )
    eeoc = _node(
        "Equal opportunity without discrimination on religious dress and grooming practices.",
        section="C. Harassment",
        node_id="eeoc",
        page=7,
    )
    nodes = [eeoc, dress]
    result = reorder_context_for_policy_topic(nodes, "What is the dress code policy?")
    assert result[0].node.node_id == "dress"