"""Per-answer trust and latency panel."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.prompts import LOW_CONFIDENCE_MESSAGE


def citation_quality_summary(citations: list[dict[str, Any]]) -> tuple[int, int]:
    cited = sum(1 for c in citations if c.get("selection_reason") == "cited_in_answer")
    fallback = sum(
        1 for c in citations if c.get("selection_reason") == "score_threshold_fallback"
    )
    return cited, fallback


def render_trust_panel(
    *,
    timing: dict[str, float] | None,
    citations: list[dict[str, Any]],
    answer: str,
    grounding_mode: str | None = None,
) -> None:
    cited, fallback = citation_quality_summary(citations)
    low_conf = LOW_CONFIDENCE_MESSAGE in answer

    with st.expander("Trust & performance", expanded=False):
        if grounding_mode == "strict":
            st.caption("Strict grounding — answers may abstain more often.")

        if low_conf:
            st.warning("Low-confidence answer — review sources carefully.")

        if timing:
            cols = st.columns(4)
            cols[0].metric("E2E", f"{timing.get('e2e_ms', 0):.0f} ms")
            cols[1].metric("Retrieve", f"{timing.get('retrieve_total_ms', 0):.0f} ms")
            cols[2].metric("Generate", f"{timing.get('generation_ms', 0):.0f} ms")
            cols[3].metric("Guard", f"{timing.get('faithfulness_guard_ms', 0):.0f} ms")

        if citations:
            st.caption(
                f"Sources: {cited} cited in answer"
                + (f", {fallback} score fallback" if fallback else "")
            )
            if fallback and not cited:
                st.info(
                    "Answer had no [Source N] tags — showing high-relevance fallback sources."
                )