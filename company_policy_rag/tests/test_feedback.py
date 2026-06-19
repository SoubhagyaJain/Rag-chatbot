"""Tests for feedback storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.feedback_store import feedback_summary, record_feedback


@pytest.fixture
def feedback_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.config import settings

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(settings, "logs_dir", log_dir)
    return log_dir / "feedback.jsonl"


def test_record_feedback_appends(feedback_file: Path) -> None:
    entry = record_feedback(
        rating=1,
        question="What is the dress code?",
        answer="Business casual.",
        model="qwen2.5:7b",
        corpus_scope="policy",
        message_id="abc123",
    )
    assert entry["rating"] == 1
    assert entry["id"] == "abc123"
    assert feedback_file.is_file()
    lines = feedback_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["question"].startswith("What is")


def test_feedback_summary_counts(feedback_file: Path) -> None:
    record_feedback(rating=1, question="q1", answer="a1", model="m")
    record_feedback(rating=-1, question="q2", answer="a2", model="m", comment="wrong")
    summary = feedback_summary()
    assert summary["total"] == 2
    assert summary["up"] == 1
    assert summary["down"] == 1
    assert len(summary["recent"]) == 2