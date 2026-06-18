"""Tests for query rewrite and policy-term augmentation."""

from __future__ import annotations

from src.query_processing import (
    augment_query_with_policy_terms,
    build_multi_retrieval_queries,
    is_comprehensive_list_query,
)


def test_augment_benefits_query() -> None:
    q = "When do new employees become eligible for health benefits?"
    expanded = augment_query_with_policy_terms(q)
    assert "health insurance" in expanded.lower()
    assert "enrollment" in expanded.lower()


def test_augment_resignation_query() -> None:
    q = "What happens if I don't give notice when resigning?"
    expanded = augment_query_with_policy_terms(q)
    assert "at-will" in expanded.lower()
    assert "resignation" in expanded.lower()


def test_augment_unrelated_query_unchanged() -> None:
    q = "What is the remote work policy?"
    assert augment_query_with_policy_terms(q) == q


def test_augment_disciplinary_query() -> None:
    q = "What is the disciplinary process for policy violations?"
    expanded = augment_query_with_policy_terms(q)
    assert "disciplinary action" in expanded.lower()
    assert "termination" in expanded.lower()


def test_augment_outside_employment_query() -> None:
    q = "Can I work a second job or do outside consulting while employed here?"
    expanded = augment_query_with_policy_terms(q)
    assert "conflict of interest" in expanded.lower()
    assert "moonlighting" in expanded.lower()


def test_is_comprehensive_list_query_detects_building_blocks() -> None:
    q = (
        "List and explain the 6 building blocks of AI Agents. Pay special attention to "
        "Role-playing, Tools (custom tools + MCP), and Memory."
    )
    assert is_comprehensive_list_query(q) is True


def test_build_multi_retrieval_queries_extracts_topics() -> None:
    q = (
        "List and explain the 6 building blocks of AI Agents. Pay special attention to "
        "Role-playing, Guardrails, and Memory (short-term, long-term)."
    )
    queries = build_multi_retrieval_queries(q, max_queries=8)
    assert q in queries
    assert any("Role-playing" in item for item in queries)
    assert any("Guardrails" in item for item in queries)
    assert any("building block 1" in item.lower() for item in queries)