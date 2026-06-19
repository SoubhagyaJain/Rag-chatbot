"""Tests for code chunk injection into retrieval candidates."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.code_retrieval import clear_code_node_cache, inject_code_chunks, promote_code_tool_nodes
from src.config import settings


def _node(text: str, *, content_type: str = "prose", node_id: str = "n1") -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, metadata={"content_type": content_type}, id_=node_id),
        score=0.5,
    )


def test_inject_code_chunks_skips_non_code_query() -> None:
    nodes = [_node("prose only")]
    result = inject_code_chunks(nodes, "What types of memory do agents use?")
    assert result == nodes


def test_inject_code_chunks_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_code_retrieval_boost", False)
    nodes = [_node("prose only")]
    result = inject_code_chunks(nodes, "Show currency conversion tool example")
    assert result == nodes


def test_inject_code_chunks_prepends_when_pool_empty(monkeypatch) -> None:
    clear_code_node_cache()

    code_node = TextNode(
        text="def convert_currency(amount, from_curr, to_curr): pass",
        metadata={"content_type": "code"},
        id_="code-1",
    )

    def _fake_code_nodes() -> list[NodeWithScore]:
        return [NodeWithScore(node=code_node, score=0.0)]

    monkeypatch.setattr("src.code_retrieval.get_guidebook_code_nodes", _fake_code_nodes)

    prose = [_node("Kayak tool prose", node_id="prose-1")]
    result = inject_code_chunks(
        prose,
        "Show the currency conversion tool example and explain how it is invoked.",
    )
    assert len(result) == 2
    assert (result[0].node.metadata or {}).get("content_type") == "code"
    assert result[0].score >= 10.0


def test_promote_currency_puts_prose_first_not_unrelated_code() -> None:
    currency_prose = _node(
        "real-time currency conversion tool live exchange rates from external API",
        node_id="curr-prose",
    )
    unrelated_code = _node(
        "def unrelated(): pass",
        content_type="code",
        node_id="other-code",
    )
    nodes = [unrelated_code, currency_prose]
    result = promote_code_tool_nodes(
        nodes,
        "What real-world capability does the currency tool demonstrate?",
    )
    assert result[0].node.node_id == "curr-prose"