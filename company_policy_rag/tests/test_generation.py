"""Tests for grounded response synthesis (no Ollama required)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.generation import (
    GroundedCompactAndRefine,
    _parse_guard_verdict,
    _preserve_balanced_partial_answer,
    apply_faithfulness_guard,
    build_grounded_response_synthesizer,
    generate_grounded_answer_with_trace,
    normalize_balanced_answer,
)
from src.prompts import INSUFFICIENT_INFO_MESSAGE, LOW_CONFIDENCE_MESSAGE, PARTIAL_ANSWER_PREFIX


def test_build_grounded_response_synthesizer() -> None:
    synth = build_grounded_response_synthesizer(llm=MagicMock())
    assert isinstance(synth, GroundedCompactAndRefine)
    assert synth._empty_response == INSUFFICIENT_INFO_MESSAGE


def test_empty_nodes_returns_abstention() -> None:
    synth = build_grounded_response_synthesizer(llm=MagicMock())
    response = synth.synthesize(QueryBundle(query_str="test"), [])
    assert INSUFFICIENT_INFO_MESSAGE in str(response)


def test_balanced_guard_passes_supported() -> None:
    assert _parse_guard_verdict("SUPPORTED", "balanced") is True
    assert _parse_guard_verdict("UNSUPPORTED", "balanced") is False


def test_strict_guard_passes_yes_only() -> None:
    assert _parse_guard_verdict("YES", "strict") is True
    assert _parse_guard_verdict("NO", "strict") is False


def test_balanced_guard_keeps_helpful_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.generation.settings.enable_faithfulness_check", True
    )
    monkeypatch.setattr(
        "src.generation.settings.faithfulness_guard_mode", "balanced"
    )
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SUPPORTED"
    nodes = [
        NodeWithScore(
            node=TextNode(text="Report harassment to your supervisor."),
            score=0.9,
        )
    ]
    answer = "Based on the available information, report harassment to your supervisor [Source 1]."
    result = apply_faithfulness_guard(answer, nodes, mock_llm)
    assert result == answer


def test_preserve_balanced_partial_strips_double_ending() -> None:
    partial = (
        f"{PARTIAL_ANSWER_PREFIX} employment is at-will [Source 1]. "
        "You may resign at any time without giving notice. "
        "The excerpts do not describe penalties for resigning without notice."
    )
    double_ended = f"{partial}\n\n{INSUFFICIENT_INFO_MESSAGE}"
    stripped = _preserve_balanced_partial_answer(double_ended)
    assert INSUFFICIENT_INFO_MESSAGE not in stripped
    assert stripped == partial


def test_preserve_balanced_partial_keeps_pure_abstention() -> None:
    assert _preserve_balanced_partial_answer(INSUFFICIENT_INFO_MESSAGE) == INSUFFICIENT_INFO_MESSAGE


def test_balanced_guard_strips_double_ending_before_check(monkeypatch) -> None:
    monkeypatch.setattr("src.generation.settings.enable_faithfulness_check", True)
    monkeypatch.setattr("src.generation.settings.faithfulness_guard_mode", "balanced")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "SUPPORTED"
    nodes = [
        NodeWithScore(
            node=TextNode(text="Employment is at-will; employees may quit at any time."),
            score=0.9,
        )
    ]
    partial = (
        f"{PARTIAL_ANSWER_PREFIX} employment is at-will [Source 1]. "
        "You may resign at any time. The excerpts do not list resignation penalties."
    )
    double_ended = f"{partial}\n\n{INSUFFICIENT_INFO_MESSAGE}"
    result = apply_faithfulness_guard(double_ended, nodes, mock_llm)
    assert INSUFFICIENT_INFO_MESSAGE not in result
    assert result == partial
    mock_llm.complete.assert_called_once()


def test_balanced_guard_rejection_keeps_original_answer(monkeypatch) -> None:
    monkeypatch.setattr("src.generation.settings.enable_faithfulness_check", True)
    monkeypatch.setattr("src.generation.settings.faithfulness_guard_mode", "balanced")
    monkeypatch.setattr("src.generation.settings.faithfulness_guard_reject_action", "keep")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "UNSUPPORTED"
    nodes = [
        NodeWithScore(
            node=TextNode(text="Employees receive three days of paid sick leave."),
            score=0.9,
        )
    ]
    answer = (
        "Based on the available information in the documents, employees receive "
        "three days of paid sick leave after 120 days [Source 1]."
    )
    result = apply_faithfulness_guard(answer, nodes, mock_llm)
    assert result == answer
    assert INSUFFICIENT_INFO_MESSAGE not in result


def test_normalize_strips_quasi_abstention_suffix() -> None:
    partial = (
        f"{PARTIAL_ANSWER_PREFIX} employment is at-will [Source 1]. "
        "You may resign at any time without giving notice. "
        "The excerpts do not describe specific penalties for resigning without notice."
    )
    quasi = (
        f"{partial}\n\nThe provided documents do not contain sufficient information "
        "to specify penalties for failing to provide resignation notice."
    )
    result = normalize_balanced_answer(quasi)
    assert "do not contain sufficient information" not in result
    assert result == partial


def test_normalize_strips_inverted_abstention_first() -> None:
    inverted = (
        f"{INSUFFICIENT_INFO_MESSAGE} The excerpts focus on electronic communications.\n\n"
        f"{PARTIAL_ANSWER_PREFIX} the excerpts do not mention remote work [Source 1]."
    )
    result = normalize_balanced_answer(inverted)
    assert not result.startswith(INSUFFICIENT_INFO_MESSAGE)
    assert "remote work" in result.lower()


def test_generate_trace_empty_nodes_abstains() -> None:
    trace = generate_grounded_answer_with_trace("test", [], MagicMock())
    assert trace.final_answer == INSUFFICIENT_INFO_MESSAGE
    assert trace.fallback_reason == "none"
    assert not trace.code_validation.triggered


def test_generate_trace_code_validation_fallback(monkeypatch) -> None:
    monkeypatch.setattr("src.generation.settings.enable_faithfulness_check", False)
    monkeypatch.setattr("src.code_validation.settings.enable_code_validation", True)
    monkeypatch.setattr("src.code_validation.settings.code_validation_use_heuristic", True)
    monkeypatch.setattr("src.code_validation.settings.enable_code_self_correction", False)

    mock_synth = MagicMock()
    mock_synth.get_response.return_value = "```python\ndef invented():\n    return 0\n```"
    mock_synth._llm = MagicMock()
    monkeypatch.setattr(
        "src.generation.build_grounded_response_synthesizer",
        lambda llm=None: mock_synth,
    )

    nodes = [
        NodeWithScore(
            node=TextNode(
                text="```python\ndef real():\n    return 1\n```",
                metadata={"content_type": "code"},
            ),
            score=0.9,
        )
    ]
    trace = generate_grounded_answer_with_trace("show code", nodes, MagicMock())
    assert trace.final_answer == LOW_CONFIDENCE_MESSAGE
    assert trace.fallback_reason == "code_validation"
    assert trace.code_validation.triggered
    assert trace.code_validation.fallback_applied


def test_balanced_guard_rejection_trims_claims(monkeypatch) -> None:
    monkeypatch.setattr("src.generation.settings.enable_faithfulness_check", True)
    monkeypatch.setattr("src.generation.settings.faithfulness_guard_mode", "balanced")
    monkeypatch.setattr("src.generation.settings.faithfulness_guard_reject_action", "trim")
    mock_llm = MagicMock()
    trimmed = (
        "Based on the available information in the documents, employees receive "
        "three days of paid sick leave after 120 days [Source 1]."
    )
    mock_llm.complete.side_effect = ["UNSUPPORTED", trimmed]
    nodes = [
        NodeWithScore(
            node=TextNode(text="Employees receive three days of paid sick leave."),
            score=0.9,
        )
    ]
    answer = (
        f"{trimmed} They also receive unlimited vacation and a company car [Source 1]."
    )
    result = apply_faithfulness_guard(answer, nodes, mock_llm)
    assert result == trimmed
    assert mock_llm.complete.call_count == 2


def test_strict_guard_rejection_abstains(monkeypatch) -> None:
    monkeypatch.setattr("src.generation.settings.enable_faithfulness_check", True)
    monkeypatch.setattr("src.generation.settings.faithfulness_guard_mode", "strict")
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "NO"
    nodes = [
        NodeWithScore(
            node=TextNode(text="Employees receive three days of paid sick leave."),
            score=0.9,
        )
    ]
    answer = "Employees receive ten days of paid sick leave."
    result = apply_faithfulness_guard(answer, nodes, mock_llm)
    assert result == INSUFFICIENT_INFO_MESSAGE