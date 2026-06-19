"""Chat welcome, history, and ChatGPT-style turn handling."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from src.agent import AgentTurnResult
from src.config import settings
from src.prompts import LOW_CONFIDENCE_MESSAGE, resolve_grounding_mode

from app.ui.components.citations import render_sources_compact
from app.ui.components.trust import render_trust_panel
from app.ui.session import index_health, run_agent_turn, run_direct_turn

SUGGESTED_PROMPTS = [
    "How many sick days do I get?",
    "What is the dress code policy?",
    "What are the six building blocks of AI agents?",
    "What is the vacation benefits policy?",
]

TYPING_INDICATOR_HTML = (
    '<div class="typing-indicator" aria-label="Generating answer">'
    "<span></span><span></span><span></span></div>"
)


def stream_answer_chunks(text: str, *, words_per_chunk: int = 3) -> Iterator[str]:
    """Yield answer text in small chunks for st.write_stream."""
    words = text.split()
    if not words:
        if text:
            yield text
        return
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        if i + words_per_chunk < len(words):
            chunk += " "
        yield chunk


def apply_queue_user_prompt(session: dict[str, Any], prompt: str) -> None:
    """Append user message and defer generation (testable without Streamlit)."""
    session.setdefault("messages", []).append({"role": "user", "content": prompt})
    session["pending_user_prompt"] = prompt


def apply_complete_assistant_turn(
    session: dict[str, Any],
    turn: AgentTurnResult,
    *,
    user_prompt: str,
) -> None:
    """Persist assistant message and clear pending state."""
    session.setdefault("messages", []).append(
        {
            "role": "assistant",
            "content": turn.answer,
            "citations": turn.citations,
            "timing": turn.timing,
            "grounding_mode": turn.grounding_mode,
            "low_confidence": turn.low_confidence,
            "user_prompt": user_prompt,
        }
    )
    session["pending_user_prompt"] = None


def queue_user_prompt(prompt: str) -> None:
    apply_queue_user_prompt(st.session_state, prompt)


def complete_assistant_turn(turn: AgentTurnResult, *, user_prompt: str) -> None:
    apply_complete_assistant_turn(st.session_state, turn, user_prompt=user_prompt)


def render_welcome() -> None:
    if st.session_state.messages:
        return

    health = index_health()
    st.markdown(
        """
        <div class="welcome-hero">
          <h3>Ask about handbook policies or guidebook content</h3>
          <p>Answers are grounded in indexed PDFs. Sources appear below each reply.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric("Indexed chunks", health.get("count", 0))
    cols[1].metric("Grounding", resolve_grounding_mode().title())
    cols[2].metric("Citations", "On" if settings.show_citations else "Off")


def render_suggested_prompts() -> None:
    if st.session_state.messages:
        return
    st.caption("Try a suggested question:")
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
                queue_user_prompt(prompt)
                st.rerun()


def _render_assistant_extras(msg: dict[str, Any]) -> None:
    citations = msg.get("citations") or []
    if settings.show_citations and citations:
        render_sources_compact(citations)
    render_trust_panel(
        timing=msg.get("timing"),
        citations=citations,
        answer=msg.get("content", ""),
        grounding_mode=msg.get("grounding_mode"),
        expanded=bool(msg.get("low_confidence")),
    )


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] != "assistant":
                continue
            _render_assistant_extras(msg)


def _run_turn(prompt: str, agent, memory, query_engine) -> AgentTurnResult:
    chat_mode = st.session_state.get("chat_mode", "direct")
    if chat_mode == "agent":
        return run_agent_turn(agent, prompt, memory)
    return run_direct_turn(query_engine, prompt)


def process_pending_turn(prompt: str, agent, memory, query_engine) -> None:
    """Generate assistant reply for the queued user prompt."""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(TYPING_INDICATOR_HTML, unsafe_allow_html=True)
        try:
            turn = _run_turn(prompt, agent, memory, query_engine)
        except Exception as exc:
            turn = AgentTurnResult(
                answer=f"Sorry, something went wrong: {exc}",
                citations=[],
                timing=None,
                grounding_mode=resolve_grounding_mode(),
                low_confidence=False,
            )
        placeholder.empty()
        st.write_stream(stream_answer_chunks(turn.answer))
        if turn.low_confidence or LOW_CONFIDENCE_MESSAGE in turn.answer:
            st.caption("⚠ Review cited sources — answer could not be fully verified.")
        if settings.show_citations and turn.citations:
            render_sources_compact(turn.citations)
        render_trust_panel(
            timing=turn.timing,
            citations=turn.citations,
            answer=turn.answer,
            grounding_mode=turn.grounding_mode,
            expanded=turn.low_confidence,
        )

    complete_assistant_turn(turn, user_prompt=prompt)
    st.rerun()


def render_chat_interface(agent, memory, query_engine) -> None:
    """Main chat loop: history, pending generation, and input."""
    st.markdown('<div class="chat-thread">', unsafe_allow_html=True)

    in_conversation = bool(st.session_state.messages)
    if not in_conversation:
        render_welcome()
        render_suggested_prompts()

    render_chat_history()

    pending = st.session_state.get("pending_user_prompt")
    if pending:
        process_pending_turn(pending, agent, memory, query_engine)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if prompt := st.chat_input("Ask a policy question…"):
        queue_user_prompt(prompt)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)