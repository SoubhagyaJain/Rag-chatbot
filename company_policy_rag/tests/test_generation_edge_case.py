"""Tests for guidebook edge-case abstention normalization."""

from __future__ import annotations

from src.generation import normalize_balanced_answer
from src.prompts import INSUFFICIENT_INFO_MESSAGE


def test_edge_case_keeps_abstention_only() -> None:
    query = (
        "How many vacation days do nonprofit employees accrue "
        "per the AI Agents guidebook?"
    )
    answer = (
        f"{INSUFFICIENT_INFO_MESSAGE}\n\n"
        "Based on the available information in the documents, "
        "there is no mention of vacation days."
    )
    result = normalize_balanced_answer(answer, query=query)
    assert result.startswith(INSUFFICIENT_INFO_MESSAGE)
    assert "Based on the available information" not in result


def test_edge_case_collapses_partial_to_abstention() -> None:
    query = (
        "How many vacation days do nonprofit employees accrue "
        "per the AI Agents guidebook?"
    )
    answer = (
        "Based on the available information in the documents, "
        "there is no mention of vacation days or nonprofit policies."
    )
    result = normalize_balanced_answer(answer, query=query)
    assert INSUFFICIENT_INFO_MESSAGE in result
    assert "Based on the available information" not in result