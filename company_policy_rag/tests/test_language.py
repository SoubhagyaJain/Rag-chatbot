"""Tests for response language helpers."""

from __future__ import annotations

from src.language import (
    append_language_hint,
    detect_query_language,
    language_instruction_for_query,
    resolve_response_language,
)


def test_detect_english_query() -> None:
    q = "List and explain the 6 building blocks of AI Agents."
    assert detect_query_language(q) == "english"


def test_detect_chinese_query() -> None:
    q = "请列出人工智能代理的六个构建模块并解释。"
    assert detect_query_language(q) == "chinese"


def test_default_response_language_is_english() -> None:
    q = "What is vacation policy?"
    assert resolve_response_language(q) == "english"
    assert "English" in language_instruction_for_query(q)
    assert "来源" in language_instruction_for_query(q)


def test_append_language_hint_adds_instruction() -> None:
    q = "Explain guardrails for AI agents."
    hinted = append_language_hint(q)
    assert q in hinted
    assert "[LANGUAGE:" in hinted