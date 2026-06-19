"""System health helpers (pure functions for tests + Streamlit page)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ollama_client import probe_ollama_tags as _probe_ollama_tags


def load_last_eval_run(results_path: Path) -> dict[str, Any] | None:
    """Return the most recent run from evaluation_results.json, or None."""
    if not results_path.is_file():
        return None
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    last = runs[-1]
    return last if isinstance(last, dict) else None


def format_eval_metrics(run: dict[str, Any] | None) -> dict[str, str]:
    """Extract display-friendly aggregate metrics from an eval run."""
    if not run:
        return {}
    aggregate = run.get("aggregate") or {}
    if not isinstance(aggregate, dict):
        return {}
    labels = {
        "answer_relevancy": "Answer relevancy",
        "faithfulness": "Faithfulness",
        "context_precision": "Context precision",
        "hit_rate": "Hit rate",
    }
    out: dict[str, str] = {}
    for key, label in labels.items():
        value = aggregate.get(key)
        if value is not None:
            out[label] = f"{float(value):.3f}"
    return out


def probe_ollama_tags(base_url: str, *, timeout: float = 5.0) -> tuple[bool, list[str], str | None]:
    """Call Ollama GET /api/tags. Returns (ok, model_names, error_message)."""
    return _probe_ollama_tags(base_url, timeout=timeout)