"""Chat welcome, history, and suggested prompts."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import settings
from src.prompts import LOW_CONFIDENCE_MESSAGE, resolve_grounding_mode

from app.ui.components.citations import render_sources
from app.ui.components.trust import render_trust_panel
from app.ui.session import index_health, run_agent_turn

SUGGESTED_PROMPTS = [
    "How many sick days do I get?",
    "What is the dress code policy?",
    "What are the six building blocks of AI agents?",
    "What is the vacation benefits policy?",
]


def render_welcome() -> None:
    if st.session_state.messages:
        return

    health = index_health()
    st.markdown(
        """
        <div class="welcome-hero">
          <h3>Ask about handbook policies or guidebook content</h3>
          <p>Answers are grounded in indexed PDFs. Sources shown below each reply match what grounded the answer — not a parallel search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric("Indexed chunks", health.get("count", 0))
    cols[1].metric("Grounding", resolve_grounding_mode().title())
    cols[2].metric("Citations", "On" if settings.show_citations else "Off")


def render_suggested_prompts(on_select) -> None:
    if st.session_state.messages:
        return
    st.caption("Try a suggested question:")
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
                on_select(prompt)


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] != "assistant":
                continue
            if msg.get("low_confidence") or LOW_CONFIDENCE_MESSAGE in msg.get("content", ""):
                st.info(
                    "This answer could not be fully verified against retrieved sources. "
                    "Review the cited excerpts below or ask a more specific question."
                )
            citations = msg.get("citations") or []
            if settings.show_citations and citations:
                render_sources(citations)
            render_trust_panel(
                timing=msg.get("timing"),
                citations=citations,
                answer=msg.get("content", ""),
                grounding_mode=msg.get("grounding_mode"),
            )


def handle_user_message(prompt: str, agent, memory) -> None:
    """Run one turn, append to session history, and rerun to render via history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.status("Searching policies and generating answer…", expanded=True):
        try:
            turn = run_agent_turn(agent, prompt, memory)
            answer = turn.answer
            citations = turn.citations
            timing = turn.timing
            grounding_mode = turn.grounding_mode
            low_confidence = turn.low_confidence
        except Exception as exc:
            answer = f"Sorry, something went wrong: {exc}"
            citations = []
            timing = None
            grounding_mode = resolve_grounding_mode()
            low_confidence = False

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "timing": timing,
            "grounding_mode": grounding_mode,
            "low_confidence": low_confidence,
        }
    )
    st.rerun()