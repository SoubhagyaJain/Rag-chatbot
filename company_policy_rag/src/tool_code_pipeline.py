"""Lightweight Tool/Code pipeline: query classification and context reordering."""

from __future__ import annotations

from enum import Enum

from llama_index.core.schema import NodeWithScore

from src.config import settings
from src.query_processing import is_code_or_tool_query

_CURRENCY_QUERY_MARKERS = (
    "currency",
    "convert_currency",
    "conversion tool",
    "currencyconverter",
    "exchange rate",
)

_CURRENCY_TEXT_MARKERS = (
    "currencyconverter",
    "currency conversion",
    "convert_currency",
    "exchange rate",
    "currency_analyst",
    "real-time currency",
    "live exchange",
)

_CODE_LINK_MARKERS = (
    "code is available",
    "check this code",
    "dailydoseofds.com/p/",
)

_WALKTHROUGH_MARKERS = (
    "check this out",
)

_CUSTOM_TOOL_MARKERS = (
    "custom tools via mcp",
    "build custom tools",
    "custom tools at times",
    "@mcp.tool",
    "mcp.tool()",
)


class ToolCodeKind(str, Enum):
    NONE = "none"
    CURRENCY = "currency"
    CODE_LINKS = "code_links"
    WALKTHROUGH = "walkthrough"
    GENERIC = "generic"


def classify_tool_code_query(query: str) -> ToolCodeKind:
    """Classify code/tool queries for context reordering and generation hints."""
    if not query.strip() or not is_code_or_tool_query(query):
        return ToolCodeKind.NONE

    q = query.lower()
    if any(marker in q for marker in _CURRENCY_QUERY_MARKERS):
        return ToolCodeKind.CURRENCY
    if "code is available" in q or "full code" in q or ("point" in q and "code" in q):
        return ToolCodeKind.CODE_LINKS
    if "check this out" in q or "walkthrough" in q:
        return ToolCodeKind.WALKTHROUGH
    return ToolCodeKind.GENERIC


def _node_text(nws: NodeWithScore) -> str:
    return (nws.node.get_content() or "").lower()


def _is_code_node(nws: NodeWithScore) -> bool:
    return (nws.node.metadata or {}).get("content_type") == "code"


def _matches_markers(nws: NodeWithScore, markers: tuple[str, ...]) -> bool:
    text = _node_text(nws)
    return any(marker in text for marker in markers)


def _promote_nodes_with_markers(
    nodes: list[NodeWithScore],
    markers: tuple[str, ...],
) -> list[NodeWithScore]:
    if not nodes or not markers:
        return nodes
    primary: list[NodeWithScore] = []
    rest: list[NodeWithScore] = []
    for nws in nodes:
        if _matches_markers(nws, markers):
            primary.append(nws)
        else:
            rest.append(nws)
    return primary + rest if primary else nodes


def _reorder_currency_nodes(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Prose currency chunks first; only on-topic code (not unrelated guidebook code)."""
    topic_prose = [
        n for n in nodes if not _is_code_node(n) and _matches_markers(n, _CURRENCY_TEXT_MARKERS)
    ]
    topic_code = [
        n for n in nodes if _is_code_node(n) and _matches_markers(n, _CURRENCY_TEXT_MARKERS)
    ]
    primary_ids = {n.node.node_id for n in topic_prose + topic_code if n.node.node_id}
    rest = [n for n in nodes if n.node.node_id not in primary_ids]
    if topic_prose or topic_code:
        return topic_prose + topic_code + rest
    return nodes


def _reorder_with_code_first_matching(
    nodes: list[NodeWithScore],
    markers: tuple[str, ...],
) -> list[NodeWithScore]:
    """Promote marker-matching nodes; put matching code before other matches."""
    nodes = _promote_nodes_with_markers(nodes, markers)
    topic_code = [n for n in nodes if _is_code_node(n) and _matches_markers(n, markers)]
    if not topic_code:
        return nodes
    topic_code_ids = {n.node.node_id for n in topic_code if n.node.node_id}
    non_matching_code = [
        n for n in nodes if _is_code_node(n) and n.node.node_id not in topic_code_ids
    ]
    non_code = [n for n in nodes if not _is_code_node(n)]
    primary = [n for n in nodes if _matches_markers(n, markers) and not _is_code_node(n)]
    rest_non_code = [n for n in non_code if n not in primary]
    return primary + topic_code + rest_non_code + non_matching_code


def reorder_context_for_tool_code(
    nodes: list[NodeWithScore],
    query: str,
) -> list[NodeWithScore]:
    """
    Reorder retrieved chunks so Source 1 aligns with the query topic.

    Currency queries keep prose first (p.14–17) — never blanket code_first, which
    pushed unrelated code chunks (e.g. p.46) ahead of currency prose.
    """
    if not settings.enable_code_retrieval_boost or not nodes:
        return nodes

    kind = classify_tool_code_query(query)
    if kind == ToolCodeKind.NONE:
        return nodes

    if kind == ToolCodeKind.CURRENCY:
        return _reorder_currency_nodes(nodes)

    if kind == ToolCodeKind.CODE_LINKS:
        return _reorder_with_code_first_matching(nodes, _CODE_LINK_MARKERS)

    if kind == ToolCodeKind.WALKTHROUGH:
        return _reorder_with_code_first_matching(nodes, _WALKTHROUGH_MARKERS)

    q = query.lower()
    if "custom tool" in q or "build custom" in q:
        return _promote_nodes_with_markers(nodes, _CUSTOM_TOOL_MARKERS)

    code_first = [n for n in nodes if _is_code_node(n)]
    non_code = [n for n in nodes if not _is_code_node(n)]
    if code_first:
        return code_first + non_code
    return nodes