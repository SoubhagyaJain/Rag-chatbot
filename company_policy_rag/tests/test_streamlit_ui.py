"""Pure-helper tests for Streamlit UI modules (no Streamlit runtime)."""

from __future__ import annotations

import json
from pathlib import Path

from app.ui.components.chat import SUGGESTED_PROMPTS
from app.ui.components.health import format_eval_metrics, load_last_eval_run, probe_ollama_tags
from app.ui.components.trust import citation_quality_summary
from app.ui.session import corpus_scope_filters


def test_suggested_prompts_non_empty_and_bounded():
    assert len(SUGGESTED_PROMPTS) >= 3
    assert all(isinstance(p, str) and len(p) > 10 for p in SUGGESTED_PROMPTS)


def test_citation_quality_summary():
    citations = [
        {"selection_reason": "cited_in_answer"},
        {"selection_reason": "cited_in_answer"},
        {"selection_reason": "score_threshold_fallback"},
    ]
    cited, fallback = citation_quality_summary(citations)
    assert cited == 2
    assert fallback == 1


def test_corpus_scope_filters_all_returns_none():
    assert corpus_scope_filters("all") is None
    assert corpus_scope_filters(None) is None


def test_corpus_scope_filters_policy_returns_metadata():
    filters = corpus_scope_filters("policy")
    assert filters is not None
    assert "source_file" in filters


def test_load_last_eval_run_missing_file(tmp_path: Path):
    assert load_last_eval_run(tmp_path / "missing.json") is None


def test_load_last_eval_run_returns_last(tmp_path: Path):
    path = tmp_path / "evaluation_results.json"
    path.write_text(
        json.dumps({"runs": [{"run_id": "a"}, {"run_id": "b", "aggregate": {"faithfulness": 0.8}}]}),
        encoding="utf-8",
    )
    last = load_last_eval_run(path)
    assert last is not None
    assert last["run_id"] == "b"


def test_format_eval_metrics():
    metrics = format_eval_metrics(
        {"aggregate": {"faithfulness": 0.807, "answer_relevancy": 0.766, "hit_rate": 0.886}}
    )
    assert metrics["Faithfulness"] == "0.807"
    assert metrics["Answer relevancy"] == "0.766"
    assert metrics["Hit rate"] == "0.886"


def test_probe_ollama_tags_invalid_host():
    ok, models, err = probe_ollama_tags("http://127.0.0.1:1", timeout=0.5)
    assert not ok
    assert models == []
    assert err