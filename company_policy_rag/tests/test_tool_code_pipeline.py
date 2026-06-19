"""Tests for tool/code query classification and context reordering."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.tool_code_pipeline import (
    ToolCodeKind,
    classify_tool_code_query,
    reorder_context_for_tool_code,
)


def _node(
    text: str,
    *,
    content_type: str = "prose",
    node_id: str,
    page: int | None = None,
) -> NodeWithScore:
    meta: dict[str, str | int] = {"content_type": content_type}
    if page is not None:
        meta["page_number"] = page
    return NodeWithScore(
        node=TextNode(text=text, metadata=meta, id_=node_id),
        score=0.5,
    )


def test_classify_currency_query() -> None:
    kind = classify_tool_code_query(
        "Show the currency conversion tool example and explain how it is invoked."
    )
    assert kind == ToolCodeKind.CURRENCY


def test_classify_non_tool_query() -> None:
    assert classify_tool_code_query("How many sick days do employees receive?") == ToolCodeKind.NONE


def test_currency_reorder_puts_prose_before_unrelated_code() -> None:
    unrelated_code = _node(
        "def unrelated_helper():\n    return 1",
        content_type="code",
        node_id="code-46",
        page=46,
    )
    currency_prose = _node(
        "A real-time currency conversion tool fetches live exchange rates from an external API.",
        node_id="prose-14",
        page=14,
    )
    other_prose = _node("Kayak search tool example.", node_id="prose-30", page=30)

    nodes = [unrelated_code, other_prose, currency_prose]
    result = reorder_context_for_tool_code(
        nodes,
        "What real-world capability does the currency tool demonstrate?",
    )

    assert result[0].node.node_id == "prose-14"
    assert (result[0].node.metadata or {}).get("content_type") == "prose"
    assert result[0].node.node_id != "code-46"


def test_currency_reorder_keeps_matching_code_after_prose() -> None:
    currency_prose = _node(
        "CurrencyConverterTool implements currency conversion.",
        node_id="prose-16",
        page=16,
    )
    currency_code = _node(
        'tools = ["convert_currency"]  # MCP server exposes convert_currency tool',
        content_type="code",
        node_id="code-20",
        page=20,
    )
    unrelated_code = _node(
        "def kayak_search(): pass",
        content_type="code",
        node_id="code-46",
        page=46,
    )

    nodes = [unrelated_code, currency_code, currency_prose]
    result = reorder_context_for_tool_code(
        nodes,
        "Show the currency conversion tool example and explain how it is invoked.",
    )

    ids = [n.node.node_id for n in result]
    assert ids.index("prose-16") < ids.index("code-20")
    assert ids.index("code-20") < ids.index("code-46")