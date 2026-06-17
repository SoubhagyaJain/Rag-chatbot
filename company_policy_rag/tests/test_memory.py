"""Tests for conversation memory helpers (no Ollama required)."""

from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from src.config import settings
from src.memory import (
    build_retrieval_query,
    create_session_memory,
    format_history_block,
    trim_memory_to_window,
)


def _fill_memory(memory: ChatMemoryBuffer, turns: int) -> None:
    for i in range(turns):
        memory.put(ChatMessage(role=MessageRole.USER, content=f"User question {i}"))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=f"Assistant answer {i}"))


def test_create_memory_when_enabled() -> None:
    mem = create_session_memory()
    assert mem is not None


def test_create_memory_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_conversation_memory", False)
    assert create_session_memory() is None


def test_trim_memory_to_window(monkeypatch) -> None:
    monkeypatch.setattr(settings, "memory_window_size", 2)
    memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
    _fill_memory(memory, 5)
    assert len(memory.get_all()) == 10
    trim_memory_to_window(memory)
    assert len(memory.get_all()) == 4  # 2 turns × 2 messages


def test_build_retrieval_query_with_context() -> None:
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    memory.put(ChatMessage(role=MessageRole.USER, content="What is the vacation policy?"))
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Full-time employees accrue 15 days of PTO annually.",
        )
    )
    query = build_retrieval_query("What about part-time?", memory)
    assert "vacation" in query.lower() or "pto" in query.lower()
    assert "part-time" in query.lower()


def test_build_retrieval_query_no_memory() -> None:
    assert build_retrieval_query("Hello", None) == "Hello"


def test_format_history_block() -> None:
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    memory.put(ChatMessage(role=MessageRole.USER, content="Test question"))
    block = format_history_block(memory)
    assert "<conversation_history>" in block
    assert "user" in block.lower()