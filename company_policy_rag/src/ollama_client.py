"""Ollama HTTP helpers shared by API, Streamlit, and chat service."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from src.config import settings

_EMBED_MARKERS = ("embed", "nomic-embed", "mxbai-embed", "bge-m3")


def filter_chat_models(names: list[str]) -> list[str]:
    """Exclude embedding-only models from LLM picker options."""
    chat: list[str] = []
    for name in names:
        lower = name.lower()
        if any(marker in lower for marker in _EMBED_MARKERS):
            continue
        chat.append(name)
    return sorted(chat)


def format_model_label(model_id: str) -> str:
    """Human-readable label for UI (qwen2.5:7b -> Qwen2.5 7B)."""
    base = model_id.split(":")[0]
    tag = model_id.split(":")[1] if ":" in model_id else ""
    label = base.replace("-", " ").replace("_", " ")
    parts = label.split()
    formatted = " ".join(p[:1].upper() + p[1:] for p in parts if p)
    if tag:
        formatted = f"{formatted} {tag.upper()}"
    return formatted.strip()


def probe_ollama_tags(
    base_url: str | None = None,
    *,
    timeout: float = 5.0,
) -> tuple[bool, list[str], str | None]:
    """Call Ollama GET /api/tags. Returns (ok, model_names, error_message)."""
    url = (base_url or settings.ollama_base_url).rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return False, [], str(exc.reason if hasattr(exc, "reason") else exc)
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, [], str(exc)

    models = payload.get("models") or []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return True, sorted(names), None