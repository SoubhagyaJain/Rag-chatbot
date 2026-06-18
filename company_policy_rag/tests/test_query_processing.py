"""Tests for query rewrite and policy-term augmentation."""

from __future__ import annotations

from src.query_processing import (
    augment_query_for_retrieval,
    augment_query_with_guidebook_terms,
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


def test_augment_guidebook_building_blocks() -> None:
    q = "List and explain the 6 building blocks of AI Agents."
    expanded = augment_query_with_guidebook_terms(q)
    assert "role-playing" in expanded.lower()
    assert "guardrails" in expanded.lower()


def test_augment_guidebook_memory_types() -> None:
    q = "What types of memory do agents use?"
    expanded = augment_query_with_guidebook_terms(q)
    assert "short-term" in expanded.lower()
    assert "long-term" in expanded.lower()


def test_augment_for_retrieval_chains_policy_and_guidebook() -> None:
    q = "What types of memory do agents use?"
    expanded = augment_query_for_retrieval(q)
    assert "short-term" in expanded.lower()


def test_is_comprehensive_list_query_detects_building_blocks() -> None:
    q = (
        "List and explain the 6 building blocks of AI Agents. Pay special attention to "
        "Role-playing, Tools (custom tools + MCP), and Memory."
    )
    assert is_comprehensive_list_query(q) is True


def test_is_comprehensive_list_query_detects_memory_types() -> None:
    q = "What types of memory do agents use?"
    assert is_comprehensive_list_query(q) is True


def test_is_comprehensive_list_query_detects_design_patterns() -> None:
    q = "What are the most popular agent design patterns mentioned?"
    assert is_comprehensive_list_query(q) is True


def test_is_comprehensive_list_query_detects_subagent_roles() -> None:
    q = "What roles can sub-agents play in orchestration?"
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
    assert any("role-playing building block" in item.lower() for item in queries)


def test_build_multi_retrieval_queries_memory_types() -> None:
    q = "What types of memory do agents use?"
    queries = build_multi_retrieval_queries(q, max_queries=8)
    assert any("short-term memory" in item.lower() for item in queries)
    assert any("long-term memory" in item.lower() for item in queries)


def test_build_multi_retrieval_queries_design_patterns() -> None:
    q = "What are the most popular agent design patterns mentioned?"
    queries = build_multi_retrieval_queries(q, max_queries=8)
    assert any("react" in item.lower() for item in queries)