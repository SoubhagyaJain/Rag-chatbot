"""
Query preprocessing to improve retrieval precision.

Underspecified or conversational queries hurt vector recall AND reranker input
quality. A lightweight LLM rewrite (local Ollama) produces a keyword-dense
search query without HyDE's hallucination risk on policy text.
"""

from __future__ import annotations

from llama_index.core import Settings

from src.config import settings
from src.timing import get_current_timing, record_stage
from src.utils import logger, timer

# Deterministic term expansion before LLM rewrite — improves recall for edge cases
# where embedding similarity misses handbook vocabulary (at-will, health insurance).
_TOPIC_EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (
        ("health benefit", "health insurance", "benefits", "eligible for health", "enrollment"),
        "health insurance medical dental vision eligibility enrollment waiting period 30 days",
    ),
    (
        ("resign", "resignation", "quit", "notice when", "two weeks", "give notice"),
        "employment at-will termination separation resignation notice period",
    ),
    (
        ("dress code", "attire", "grooming", "appearance"),
        "dress code appearance grooming professional attire",
    ),
    (
        ("confidential", "trade secret", "proprietary"),
        "confidential trade secret proprietary non-disclosure",
    ),
    (
        ("disciplinary", "discipline", "policy violation", "corrective action"),
        "disciplinary action corrective action termination violation investigation report supervisor",
    ),
    (
        ("second job", "outside consulting", "moonlight", "outside employment", "consulting while"),
        "outside employment moonlighting conflict of interest electronic communications ethics approval",
    ),
]


def augment_query_with_policy_terms(query: str) -> str:
    """
    Append handbook vocabulary when the question signals specific policy topics.

    Runs before LLM rewrite so Chroma over-retrieval sees both user terms and
    section keywords (e.g. at-will for resignation, medical insurance for benefits).
    """
    q_lower = query.lower()
    extras: list[str] = []
    for triggers, terms in _TOPIC_EXPANSIONS:
        if any(trigger in q_lower for trigger in triggers):
            extras.append(terms)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


REWRITE_PROMPT = """You rewrite employee policy questions into concise search queries for a document retrieval system.

Rules:
- Output ONLY the rewritten search query (no explanation, max 35 words).
- Include specific policy terms (e.g. vacation, PTO, harassment, benefits, sick leave).
- Map user terms to handbook vocabulary when helpful:
  * resignation / quit without notice → employment at-will termination separation
  * social media / online posts → internet electronic communications online conduct
  * second job / consulting / moonlighting → outside employment conflict of interest electronic communications ethics
- Resolve pronouns using any conversation context provided.
- Do not invent facts.

{context_block}Question: {query}
Search query:"""


def _context_block_from_query(query: str) -> tuple[str, str]:
    """Split conversation-expanded queries from build_retrieval_query."""
    marker = "Current question:"
    if marker in query:
        parts = query.split(marker, 1)
        return parts[0].strip() + "\n", parts[1].strip()
    return "", query.strip()


def rewrite_query_for_retrieval(query: str) -> str:
    """
    LLM-based query rewrite for Chroma + reranker input.

    Disabled when ENABLE_QUERY_REWRITE=false. Falls back to original query on error.
    Adds ~200–800 ms per call on local Ollama — disable for latency-sensitive A/B.
    """
    if not settings.enable_query_rewrite:
        return query

    context_block, core_question = _context_block_from_query(query)
    if not core_question:
        return query

    core_question = augment_query_with_policy_terms(core_question)

    llm = Settings.llm
    if llm is None:
        logger.warning("Query rewrite skipped — no LLM configured in Settings")
        return query

    prompt = REWRITE_PROMPT.format(
        context_block=f"Context:\n{context_block}\n" if context_block else "",
        query=core_question,
    )

    try:
        with timer("query_rewrite") as t:
            response = str(llm.complete(prompt)).strip()
        if get_current_timing() is not None:
            record_stage("query_rewrite", t["elapsed_ms"])
        # Take first line only — models sometimes add chatter
        rewritten = response.splitlines()[0].strip().strip('"').strip("'")
        if len(rewritten) < 5:
            return query
        logger.debug("Query rewrite: '%s' → '%s'", core_question[:80], rewritten[:80])
        return rewritten
    except Exception as exc:
        logger.warning("Query rewrite failed, using original: %s", exc)
        return query