"""Extract reasoning-model thinking blocks from LLM output."""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass

_thinking_this_turn: ContextVar[str | None] = ContextVar("thinking_this_turn", default=None)

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REASONING_MODEL_MARKERS = ("deepseek-r1", "deepseek-r1:", "r1:", "reasoning", "qwq")


@dataclass
class ThinkingResult:
    thinking: str | None
    answer: str


def is_reasoning_model(model_id: str) -> bool:
    lower = (model_id or "").lower()
    return any(marker in lower for marker in _REASONING_MODEL_MARKERS)


def extract_thinking(text: str) -> ThinkingResult:
    """Split `<think>...</think>` blocks from the visible answer."""
    if not text or not text.strip():
        return ThinkingResult(thinking=None, answer=text or "")

    blocks = _THINK_PATTERN.findall(text)
    cleaned = _THINK_PATTERN.sub("", text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    thinking = "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    return ThinkingResult(thinking=thinking, answer=cleaned or text.strip())


def begin_thinking_turn() -> None:
    _thinking_this_turn.set(None)


def set_thinking_this_turn(text: str | None) -> None:
    _thinking_this_turn.set(text)


def get_thinking_this_turn() -> str | None:
    return _thinking_this_turn.get()


def split_llm_response(raw: str, *, model_id: str | None = None) -> ThinkingResult:
    """Unified entry: always strip think tags; also accept Ollama-style payloads."""
    if not raw:
        return ThinkingResult(thinking=None, answer="")

    result = extract_thinking(raw)
    if result.thinking or result.answer != raw.strip():
        return result

    if model_id and is_reasoning_model(model_id):
        return result
    return ThinkingResult(thinking=None, answer=raw.strip())