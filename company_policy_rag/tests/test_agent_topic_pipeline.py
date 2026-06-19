"""Tests for agent-topic classification and reordering."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.agent_topic_pipeline import (
    AgentTopicKind,
    classify_agent_topic_query,
    reorder_context_for_agent_topic,
)


def _node(text: str, *, section: str = "", node_id: str = "n1") -> NodeWithScore:
    meta = {"section_path": section} if section else {}
    return NodeWithScore(
        node=TextNode(text=text, metadata=meta, id_=node_id),
        score=0.5,
    )


def test_classify_manager_agent() -> None:
    kind = classify_agent_topic_query(
        "What does a manager agent do in a multi-agent setup?"
    )
    assert kind == AgentTopicKind.MANAGER_AGENT


def test_classify_rag_in_agent() -> None:
    kind = classify_agent_topic_query(
        "How is RAG used inside an agent workflow?"
    )
    assert kind == AgentTopicKind.RAG_IN_AGENT


def test_classify_memory_block() -> None:
    kind = classify_agent_topic_query(
        "How does Memory work as a building block of AI agents?"
    )
    assert kind == AgentTopicKind.MEMORY_BLOCK


def test_manager_reorder_puts_definition_first() -> None:
    noise = _node(
        "Multi-agent Hotel Finder parses travel queries.",
        section="2. Browserbase tool",
        node_id="noise",
    )
    toc = _node(
        "#5) Multi-Agent pattern in table of contents only.",
        section="2025 EDITION > GUIDEBOOK",
        node_id="toc",
    )
    manager = _node(
        "A manager agent coordinates multiple sub-agents and decides next steps.",
        section="5 Levels of Agentic AI Systems",
        node_id="mgr",
    )
    result = reorder_context_for_agent_topic(
        [noise, toc, manager],
        "What does a manager agent do in a multi-agent setup?",
    )
    assert result[0].node.node_id == "mgr"


def test_memory_reorder_prefers_detail_over_overview() -> None:
    overview = _node(
        "Memory is one of six building blocks.",
        section="6. Memory",
        node_id="ov",
    )
    detail = _node(
        "Short-term memory exists during execution; long-term memory persists.",
        section="6. Memory",
        node_id="det",
    )
    result = reorder_context_for_agent_topic(
        [overview, detail],
        "How does Memory work as a building block of AI agents?",
    )
    assert result[0].node.node_id == "det"