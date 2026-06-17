"""
Short-term conversation memory for multi-turn policy Q&A.

Design (production-rag):
- ChatMemoryBuffer stores recent user/assistant turns locally per Chainlit session.
- System instructions (agent.py) stay separate from rolling chat history.
- Retrieval uses an expanded query so follow-ups like "And for part-time?" still
  hit the right handbook sections — memory improves both reasoning AND recall.
- Token cap prevents context overflow on long sessions; turn cap trims oldest exchanges.
"""

from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from src.config import settings
from src.utils import logger


def create_session_memory() -> ChatMemoryBuffer | None:
    """
    Create a fresh per-session memory buffer, or None if disabled.

    token_limit is a hard cap; memory_window_size trims by turn count afterward.
    """
    if not settings.enable_conversation_memory:
        return None

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=settings.memory_token_limit,
    )
    logger.debug(
        "Session memory created | window=%d turns | token_limit=%d",
        settings.memory_window_size,
        settings.memory_token_limit,
    )
    return memory


def get_history_messages(memory: ChatMemoryBuffer | None) -> list[ChatMessage]:
    """Return all messages in the buffer (empty if memory disabled)."""
    if memory is None:
        return []
    return list(memory.get_all())


def trim_memory_to_window(memory: ChatMemoryBuffer) -> None:
    """
    Enforce MEMORY_WINDOW_SIZE by dropping oldest turns.

    A turn = one user message + one assistant reply (2 ChatMessages).
    Runs after each exchange to keep memory lightweight and predictable.
    """
    messages = memory.get_all()
    max_messages = settings.memory_window_size * 2
    if len(messages) <= max_messages:
        return

    kept = messages[-max_messages:]
    memory.reset()
    for msg in kept:
        memory.put(msg)
    logger.debug(
        "Trimmed session memory to last %d turns (%d messages)",
        settings.memory_window_size,
        len(kept),
    )


def format_history_block(memory: ChatMemoryBuffer | None) -> str:
    """
    Structured conversation history for logging or optional prompt injection.

    Kept separate from SYSTEM_PROMPT — the ReAct agent injects memory automatically
    when memory= is set on the agent; this helper is for debugging / retrieval expansion.
    """
    messages = get_history_messages(memory)
    if not messages:
        return ""

    lines = ["<conversation_history>"]
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        content = (msg.content or "").strip().replace("\n", " ")
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content}")
    lines.append("</conversation_history>")
    return "\n".join(lines)


def build_retrieval_query(user_message: str, memory: ChatMemoryBuffer | None) -> str:
    """
    Expand the current question with recent conversation for retrieval + citations.

    Follow-up questions are often underspecified ("What about contractors?") —
    pairing them with prior turns materially improves Chroma + reranker recall
    without changing the indexed documents.
    """
    if memory is None or not settings.enable_conversation_memory:
        return user_message

    messages = get_history_messages(memory)
    if not messages:
        return user_message

    # Include up to the last 2 complete turns (4 messages) for retrieval context
    recent = messages[-(min(settings.memory_window_size, 2) * 2) :]
    if not recent:
        return user_message

    context_parts: list[str] = []
    for msg in recent:
        role = "User" if msg.role == MessageRole.USER else "Assistant"
        snippet = (msg.content or "").strip()[:300]
        if snippet:
            context_parts.append(f"{role}: {snippet}")

    if not context_parts:
        return user_message

    return (
        "Given this conversation context:\n"
        + "\n".join(context_parts)
        + f"\n\nCurrent question: {user_message}"
    )


def memory_stats(memory: ChatMemoryBuffer | None) -> dict[str, int | bool]:
    """Lightweight stats for Chainlit startup / debugging."""
    if memory is None:
        return {"enabled": False, "messages": 0, "turns": 0}
    messages = get_history_messages(memory)
    return {
        "enabled": True,
        "messages": len(messages),
        "turns": len(messages) // 2,
    }