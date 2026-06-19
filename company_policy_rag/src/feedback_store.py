"""Append-only local feedback log for answer quality signals."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import settings

Rating = Literal[1, -1]


def _feedback_path() -> Path:
    return settings.logs_dir / "feedback.jsonl"


def record_feedback(
    *,
    rating: Rating,
    question: str,
    answer: str,
    model: str,
    corpus_scope: str = "all",
    message_id: str | None = None,
    comment: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": message_id or str(uuid.uuid4())[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "question": question[:2000],
        "answer": answer[:4000],
        "model": model,
        "corpus_scope": corpus_scope,
    }
    if comment:
        entry["comment"] = comment[:500]
    if extra:
        entry["extra"] = extra

    path = _feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def feedback_summary(*, limit: int = 500) -> dict[str, Any]:
    path = _feedback_path()
    if not path.is_file():
        return {"total": 0, "up": 0, "down": 0, "recent": []}

    up = down = 0
    recent: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("rating") == 1:
                up += 1
            elif row.get("rating") == -1:
                down += 1
            recent.append(row)

    recent = recent[-limit:]
    return {"total": up + down, "up": up, "down": down, "recent": recent[-20:]}