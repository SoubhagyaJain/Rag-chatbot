"""Tests for SSE streaming helpers."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.streaming_generation import _sse, stream_chat_turn


def test_sse_format() -> None:
    line = _sse("token", "hello")
    assert line.startswith("event: token\n")
    assert "data: hello" in line


def test_stream_chat_turn_emits_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-streaming synthesis path yields retrieval_done, token(s), and done."""
    monkeypatch.setenv("IS_TESTING", "1")
    mock_node = MagicMock()
    mock_node.node.get_content.return_value = "Dress code excerpt"
    mock_node.node.metadata = {"section_path": "Dress Code", "page_number": 3}
    mock_node.score = 0.92

    with (
        patch("src.streaming_generation.record_generation_sources"),
        patch("src.streaming_generation.build_grounded_response_synthesizer") as mock_synth_cls,
        patch("src.streaming_generation.get_text_qa_template") as mock_template,
        patch("src.streaming_generation.format_nodes_for_prompt", return_value="ctx"),
        patch("src.streaming_generation._format_text_chunks", return_value=["ctx"]),
        patch("src.streaming_generation.apply_faithfulness_guard", side_effect=lambda a, *_: a),
        patch("src.streaming_generation.apply_code_validation_pipeline", side_effect=lambda _q, a, *_: (a, None)),
        patch("src.streaming_generation.normalize_balanced_answer", side_effect=lambda a, **_: a),
        patch("src.streaming_generation.select_citations_for_answer", return_value=[]),
        patch("src.streaming_generation.get_generation_nodes_this_turn", return_value=[mock_node]),
        patch("src.streaming_generation.get_current_timing", return_value=None),
        patch("src.streaming_generation.record_stage"),
        patch("src.streaming_generation.build_retrieval_trace", return_value={"chunk_count": 1}),
        patch("src.streaming_generation.clear_timing"),
        patch("src.streaming_generation.app_settings") as mock_settings,
    ):
        mock_settings.show_citations = False
        mock_settings.llm_model = "qwen2.5:7b"
        mock_settings.grounding_strictness = "balanced"

        synth = MagicMock()
        synth._llm = object()
        synth.get_response.return_value = "Business casual is required."
        mock_synth_cls.return_value = synth
        mock_template.return_value.format.return_value = "prompt"

        events = list(stream_chat_turn("dress code?", [mock_node], t0=0.0))

    event_types = [e.split("\n")[0].replace("event: ", "") for e in events if e.startswith("event:")]
    assert "retrieval_done" in event_types
    assert "done" in event_types
    assert "error" not in event_types

    done_raw = next(e for e in events if e.startswith("event: done"))
    payload = json.loads(done_raw.split("data: ", 1)[1].strip())
    assert "Business casual" in payload["answer"]
    assert "message_id" in payload