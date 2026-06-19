"""Tests for post-generation code validation (no Ollama required)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from src.code_validation import (
    HeuristicResult,
    _escape_format_braces,
    answer_contains_code,
    apply_code_validation_pipeline,
    context_contains_code,
    extract_code_lines,
    heuristic_code_grounded,
    remove_orphan_code_preambles,
    should_validate_code,
    strip_unsupported_code,
)
from src.prompts import get_code_validation_prompt
from src.prompts import LOW_CONFIDENCE_MESSAGE


def _node(text: str, **meta: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata=dict(meta)), score=0.9)


class TestPromptFormatting:
    def test_escape_format_braces_allows_code_in_validation_prompt(self) -> None:
        template = get_code_validation_prompt(mode="balanced")
        answer = "return {converted_amount}"
        prompt = template.format(
            context=_escape_format_braces("ctx"),
            answer=_escape_format_braces(answer),
        )
        assert "{converted_amount}" in prompt

    def test_failed_lines_with_braces_do_not_break_prompt_format(self) -> None:
        template = get_code_validation_prompt(
            mode="balanced",
            failed_lines=["return {from_curr}"],
        )
        prompt = template.format(
            context=_escape_format_braces("ctx"),
            answer=_escape_format_braces("answer"),
        )
        assert "{from_curr}" in prompt


class TestCodeDetection:
    def test_answer_contains_fenced_code(self) -> None:
        answer = "Use this snippet:\n```python\ndef hello():\n    pass\n```"
        assert answer_contains_code(answer)

    def test_answer_contains_code_block_prefix(self) -> None:
        assert answer_contains_code("[CODE BLOCK 1] def run(): pass")

    def test_context_contains_code_metadata(self) -> None:
        nodes = [_node("plain text", content_type="code")]
        assert context_contains_code(nodes)

    def test_should_validate_when_answer_has_code(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_trigger_mode", "answer_only"
        )
        nodes = [_node("Policy text only.")]
        assert should_validate_code("```\nimport os\n```", nodes)

    def test_should_not_validate_context_only_answer_only(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_trigger_mode", "answer_only"
        )
        nodes = [_node("```python\ndef x(): pass\n```", content_type="code")]
        answer = "Based on the documents, agents use memory and tools."
        assert not should_validate_code(answer, nodes)

    def test_should_validate_context_when_answer_or_context_mode(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_trigger_mode", "answer_or_context"
        )
        nodes = [_node("```python\ndef x(): pass\n```")]
        answer = "Prose only."
        assert should_validate_code(answer, nodes)


class TestCodeExtraction:
    def test_extract_code_lines_from_fence(self) -> None:
        text = "```python\ndef add(a, b):\n    return a + b\n```"
        lines = extract_code_lines(text)
        assert "def add(a, b):" in lines
        assert "return a + b" in lines

    def test_heuristic_passes_when_lines_in_context(self) -> None:
        answer = "```python\ndef greet():\n    print('hi')\n```"
        context = "Example:\ndef greet():\n    print('hi')"
        result = heuristic_code_grounded(answer, context)
        assert result.passed is True

    def test_heuristic_passes_with_code_block_prefix(self) -> None:
        answer = "```python\ndef convert_currency(amount, from_curr, to_curr):\n    return amount * rate\n```"
        context = (
            "[CODE BLOCK — AI_Agents_guidebook.pdf p.42]\n"
            "def convert_currency(amount, from_curr, to_curr):\n"
            "    rate = get_exchange_rate(from_curr, to_curr)\n"
            "    return amount * rate"
        )
        result = heuristic_code_grounded(answer, context)
        assert result.passed is True

    def test_heuristic_fails_when_line_missing(self) -> None:
        answer = "```python\ndef fake():\n    return 99\n```"
        context = "def real():\n    return 1"
        result = heuristic_code_grounded(answer, context)
        assert result.passed is False
        assert result.failed

    def test_heuristic_partial_ratio_pass(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_heuristic_min_ratio", 0.5
        )
        answer = "```python\ndef a():\n    pass\ndef b():\n    pass\n```"
        context = "def a():\n    pass"
        result = heuristic_code_grounded(answer, context)
        assert result.passed is True


class TestStripUnsupportedCode:
    def test_remove_orphan_code_preamble(self) -> None:
        answer = (
            "Based on the available information in the documents, the currency tool is defined as:\n"
            "The guidebook describes real-time currency conversion [Source 1]."
        )
        cleaned = remove_orphan_code_preambles(answer)
        assert "defined as" not in cleaned
        assert "real-time currency conversion" in cleaned

    def test_strips_failed_fence_keeps_prose(self) -> None:
        answer = (
            "The tool converts currency in real time.\n\n"
            "```python\ndef convert_currency():\n    pass\n```\n\n"
            "This is useful for agents."
        )
        failed = ["def convert_currency():"]
        stripped = strip_unsupported_code(answer, failed)
        assert "converts currency" in stripped
        assert "useful for agents" in stripped
        assert "```" not in stripped
        assert "convert_currency" not in stripped


class TestValidationPipeline:
    def test_skips_when_validation_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", False)
        nodes = [_node("```python\ndef x(): pass\n```", content_type="code")]
        answer = "```python\ndef x(): pass\n```"
        final, trace = apply_code_validation_pipeline("query", answer, nodes, MagicMock())
        assert final == answer
        assert not trace.triggered

    def test_passes_heuristic_without_llm_call(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr("src.code_validation.settings.code_validation_use_heuristic", True)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_trigger_mode", "answer_only"
        )
        nodes = [
            _node(
                "```python\ndef policy_fn():\n    return True\n```",
                content_type="code",
            )
        ]
        answer = "```python\ndef policy_fn():\n    return True\n```"
        mock_llm = MagicMock()
        final, trace = apply_code_validation_pipeline("query", answer, nodes, mock_llm)
        assert final == answer
        assert trace.triggered
        assert trace.passed is True
        assert trace.validation_method == "heuristic"
        mock_llm.complete.assert_not_called()

    def test_skips_prose_when_context_has_code_answer_only(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_trigger_mode", "answer_only"
        )
        nodes = [_node("```python\ndef x(): pass\n```")]
        answer = "Agents use six building blocks including memory."
        mock_llm = MagicMock()
        final, trace = apply_code_validation_pipeline("query", answer, nodes, mock_llm)
        assert final == answer
        assert not trace.triggered
        mock_llm.complete.assert_not_called()

    def test_strip_code_on_failure(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr("src.code_validation.settings.code_validation_use_heuristic", True)
        monkeypatch.setattr("src.code_validation.settings.enable_code_self_correction", False)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_fail_mode", "strip_code"
        )
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_trigger_mode", "answer_only"
        )

        nodes = [_node("Some prose about currency tools without code.")]
        answer = (
            "The currency tool demonstrates real-time conversion.\n\n"
            "```python\ndef convert_currency():\n    return 1\n```"
        )
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "VERDICT: NO\nEXPLANATION: invented function"

        final, trace = apply_code_validation_pipeline("query", answer, nodes, mock_llm)
        assert "real-time conversion" in final
        assert "```" not in final
        assert final != LOW_CONFIDENCE_MESSAGE
        assert trace.passed is True
        assert trace.validation_method == "strip_code"
        assert not trace.fallback_applied

    def test_self_correct_then_fallback_mode(self, monkeypatch) -> None:
        monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
        monkeypatch.setattr("src.code_validation.settings.code_validation_use_heuristic", True)
        monkeypatch.setattr("src.code_validation.settings.enable_code_self_correction", True)
        monkeypatch.setattr("src.code_validation.settings.code_self_correction_max_retries", 1)
        monkeypatch.setattr(
            "src.code_validation.settings.code_validation_fail_mode", "fallback"
        )

        nodes = [_node("def real():\n    return 1", content_type="code")]
        answer = "```python\ndef fake():\n    return 99\n```"
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            "VERDICT: NO\nEXPLANATION: invented function",
            "still wrong ```python\ndef fake():\n    return 99\n```",
            "VERDICT: NO\nEXPLANATION: still wrong",
        ]

        final, trace = apply_code_validation_pipeline("query", answer, nodes, mock_llm)
        assert final == LOW_CONFIDENCE_MESSAGE
        assert trace.triggered
        assert trace.self_corrected
        assert trace.fallback_applied
        assert trace.passed is False