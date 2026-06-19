"""Tests for planning-block answer normalization."""

from __future__ import annotations

from src.generation import normalize_balanced_answer


def test_planning_block_trims_elaboration() -> None:
    query = "What is the Planning building block in AI agents?"
    answer = (
        "Based on the available information in the documents:\n"
        "4. Planning — subdividing tasks and outlining objectives to solve tasks "
        "more effectively [Source 1].\n\n"
        "This planning helps break down complex tasks into manageable steps."
    )
    result = normalize_balanced_answer(answer, query=query)
    assert "subdividing tasks" in result
    assert "break down complex tasks" not in result