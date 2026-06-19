"""Tests for reasoning-model thinking extraction."""

from __future__ import annotations

from src.thinking_extract import (
    begin_thinking_turn,
    extract_thinking,
    get_thinking_this_turn,
    is_reasoning_model,
    set_thinking_this_turn,
    split_llm_response,
)


def test_extract_thinking_strips_tags() -> None:
    raw = "<think>Let me reason.</think>\n\nThe dress code requires business casual."
    result = extract_thinking(raw)
    assert result.thinking == "Let me reason."
    assert "business casual" in result.answer
    assert "<think>" not in result.answer


def test_extract_thinking_no_tags() -> None:
    result = extract_thinking("Plain answer.")
    assert result.thinking is None
    assert result.answer == "Plain answer."


def test_is_reasoning_model() -> None:
    assert is_reasoning_model("deepseek-r1:7b")
    assert is_reasoning_model("qwen2.5:7b") is False


def test_thinking_context_var() -> None:
    begin_thinking_turn()
    assert get_thinking_this_turn() is None
    set_thinking_this_turn("chain of thought")
    assert get_thinking_this_turn() == "chain of thought"
    begin_thinking_turn()
    assert get_thinking_this_turn() is None


def test_split_llm_response_with_tags() -> None:
    result = split_llm_response("<think>hidden</think>Visible", model_id="qwen2.5:7b")
    assert result.thinking == "hidden"
    assert result.answer == "Visible"