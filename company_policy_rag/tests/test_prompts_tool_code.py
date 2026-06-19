"""Regression guards for tool/code few-shot prompts."""

from __future__ import annotations

from src.prompts import FEW_SHOT_CODE_BALANCED


def test_example_k_does_not_teach_invented_convert_currency_def() -> None:
    """Example K GOOD must not show def convert_currency when excerpts are prose-only."""
    example_k = FEW_SHOT_CODE_BALANCED.split("### Example K2")[0]
    assert "def convert_currency(amount, from_curr, to_curr)" not in example_k
    assert "get_exchange_rate" not in example_k
    assert "```python" not in example_k


def test_example_k_uses_prose_currency_pattern() -> None:
    assert "CurrencyConverterTool" in FEW_SHOT_CODE_BALANCED
    assert "currency_analyst" in FEW_SHOT_CODE_BALANCED


def test_tools_real_world_example_present() -> None:
    assert "real-world capability does the currency tool demonstrate" in FEW_SHOT_CODE_BALANCED
    assert "live exchange rates" in FEW_SHOT_CODE_BALANCED