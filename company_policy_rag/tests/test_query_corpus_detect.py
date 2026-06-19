"""Tests for automatic policy vs guidebook corpus routing."""

from __future__ import annotations

from src.query_processing import detect_query_corpus
from src.retrieval_scope import resolve_query_filters


def test_detect_dress_code_as_policy():
    assert detect_query_corpus("What is the dress code policy?") == "policy"


def test_detect_building_blocks_as_guidebook():
    assert detect_query_corpus("What are the six building blocks of AI agents?") == "guidebook"


def test_detect_sick_leave_as_policy():
    assert detect_query_corpus("How many sick days do I get?") == "policy"


def test_resolve_filters_routes_dress_code_when_scope_all(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval_scope.settings.enable_corpus_scoped_retrieval", True
    )
    filters = resolve_query_filters("What is the dress code policy?", "all")
    assert filters is not None
    assert "Employee-Handbook" in filters["source_file"]


def test_resolve_filters_none_for_ambiguous_when_scope_all(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval_scope.settings.enable_corpus_scoped_retrieval", True
    )
    assert resolve_query_filters("Hello there", "all") is None