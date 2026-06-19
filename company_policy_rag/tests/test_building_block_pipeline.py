"""Tests for round-1 building-block topic classification and reordering."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.building_block_pipeline import (
    GuidebookTopicKind,
    classify_guidebook_topic_query,
    reorder_context_for_guidebook_topic,
)


def _node(text: str, *, section: str = "", node_id: str = "n1") -> NodeWithScore:
    meta = {"section_path": section} if section else {}
    return NodeWithScore(
        node=TextNode(text=text, metadata=meta, id_=node_id),
        score=0.5,
    )


def test_classify_guardrails() -> None:
    kind = classify_guidebook_topic_query(
        "What are Guardrails in AI agents and why are they used?"
    )
    assert kind == GuidebookTopicKind.GUARDRAILS


def test_classify_planning_block() -> None:
    kind = classify_guidebook_topic_query(
        "What is the Planning building block in AI agents?"
    )
    assert kind == GuidebookTopicKind.PLANNING_BLOCK


def test_guardrails_reorder_puts_topic_first() -> None:
    patterns = _node(
        "ReAct design pattern combines reflection and tool use.",
        section="5 Agentic AI Design Patterns",
        node_id="patterns",
    )
    guardrails = _node(
        "Guardrails ensure agents stay within safe boundaries and limit tool usage.",
        section="5 Levels of Agentic AI Systems",
        node_id="guard",
    )
    nodes = [patterns, guardrails]
    result = reorder_context_for_guidebook_topic(
        nodes,
        "What are Guardrails in AI agents and why are they used?",
    )
    assert result[0].node.node_id == "guard"


def test_planning_block_prefers_five_levels_over_patterns() -> None:
    patterns = _node(
        "ReAct design pattern combines reflection and tool use.",
        section="5 Agentic AI Design Patterns",
        node_id="pat",
    )
    block = _node(
        "Planning — subdividing tasks and outlining objectives.",
        section="5 Levels of Agentic AI Systems",
        node_id="block",
    )
    nodes = [patterns, block]
    result = reorder_context_for_guidebook_topic(
        nodes,
        "What is the Planning building block in AI agents?",
    )
    assert result[0].node.node_id == "block"