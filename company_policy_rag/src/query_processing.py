"""
Query preprocessing to improve retrieval precision.

Underspecified or conversational queries hurt vector recall AND reranker input
quality. A lightweight LLM rewrite (local Ollama) produces a keyword-dense
search query without HyDE's hallucination risk on policy text.
"""

from __future__ import annotations

import re

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

_GUIDEBOOK_TOPIC_EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (
        ("building block", "building blocks", "six building"),
        "Role-playing Focus Tasks Tools Cooperation Guardrails Planning Memory six AI agents",
    ),
    (
        ("types of memory", "memory do agents", "memory types", "memory agents use"),
        "short-term long-term entity episodic semantic procedural memory",
    ),
    (
        ("design pattern", "design patterns", "agent pattern", "most popular"),
        "ReAct reflection planning tool use multi-agent orchestration design patterns",
    ),
    (
        ("sub-agent", "sub-agents", "subagent", "orchestration roles", "roles can"),
        "manager specialist research summarization delegation orchestration sub-agent",
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


def augment_query_with_guidebook_terms(query: str) -> str:
    """Append guidebook vocabulary for AI-agent enumeration and topic queries."""
    q_lower = query.lower()
    extras: list[str] = []
    for triggers, terms in _GUIDEBOOK_TOPIC_EXPANSIONS:
        if any(trigger in q_lower for trigger in triggers):
            extras.append(terms)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


def augment_query_for_retrieval(query: str) -> str:
    """Deterministic term expansion before optional LLM rewrite."""
    return augment_query_with_guidebook_terms(augment_query_with_policy_terms(query))


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

    Disabled when ENABLE_QUERY_REWRITE=false. Falls back to augmented query on error.
    Adds ~200–800 ms per call on local Ollama — disable for latency-sensitive A/B.
    """
    query = augment_query_for_retrieval(query)
    if not settings.enable_query_rewrite:
        return query

    context_block, core_question = _context_block_from_query(query)
    if not core_question:
        return query

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


_COMPREHENSIVE_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blist\b.+\bexplain\b", re.IGNORECASE),
    re.compile(r"\blist\b", re.IGNORECASE),
    re.compile(r"\bbuilding\s+blocks?\b", re.IGNORECASE),
    re.compile(r"\bpay\s+special\s+attention\b", re.IGNORECASE),
    re.compile(r"\bfor\s+each\b", re.IGNORECASE),
    re.compile(r"\ball\s+\d+\b", re.IGNORECASE),
    re.compile(r"\btypes?\s+of\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+the\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\broles?\s+can\b", re.IGNORECASE),
    re.compile(r"\bdesign\s+patterns?\b", re.IGNORECASE),
    re.compile(r"\bmost\s+popular\b", re.IGNORECASE),
)

_COMPREHENSIVE_MIN_QUERY_LEN = 25


def is_comprehensive_list_query(query: str) -> bool:
    """True when the user asks for a multi-part list or enumeration from documents."""
    text = query.strip()
    if len(text) < _COMPREHENSIVE_MIN_QUERY_LEN:
        return False
    return any(pattern.search(text) for pattern in _COMPREHENSIVE_QUERY_PATTERNS)


_BUILDING_BLOCK_SUBQUERIES: tuple[str, ...] = (
    "5 Levels of Agentic AI Systems building blocks overview",
    "six building blocks Role-playing Tools Memory Guardrails Planning",
    "Role-playing building block AI agents",
    "Tools MCP building block AI agents",
    "Memory building block AI agents",
    "Guardrails building block AI agents",
    "Planning building block AI agents",
    "Cooperation Focus Tasks building block AI agents",
)

_MEMORY_TYPE_SUBQUERIES: tuple[str, ...] = (
    "short-term memory agents",
    "long-term memory agents",
    "entity memory agents",
)

_DESIGN_PATTERN_SUBQUERIES: tuple[str, ...] = (
    "ReAct agent design pattern",
    "reflection agent pattern",
    "planning pattern agents",
    "tool use agent pattern",
)

_SUBAGENT_ROLE_SUBQUERIES: tuple[str, ...] = (
    "research agent orchestration",
    "manager agent sub-agent specialization",
    "sub-agent roles delegation",
)


def _append_unique_query(queries: list[str], seen: set[str], item: str) -> None:
    key = item.lower()
    if key in seen:
        return
    seen.add(key)
    queries.append(item)


def _split_topic_clause(clause: str) -> list[str]:
    """Split comma-separated topics while keeping parenthetical hints attached."""
    parts = re.split(r",(?![^()]*\))", clause)
    topics: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" .")
        if len(cleaned) >= 3:
            topics.append(cleaned)
    return topics


def build_multi_retrieval_queries(query: str, *, max_queries: int = 8) -> list[str]:
    """
    Build supplementary retrieval queries for comprehensive list questions.

    Merges results from each query before reranking so all requested sections
    (e.g. six AI-agent building blocks) can appear in generation context.
    """
    core = query.strip()
    if not core:
        return []

    queries: list[str] = []
    seen: set[str] = set()
    _append_unique_query(queries, seen, core)

    q_lower = core.lower()

    if re.search(r"building\s+blocks?", q_lower):
        _append_unique_query(queries, seen, "six building blocks overview AI agents")
        for subquery in _BUILDING_BLOCK_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"how\s+many", q_lower) and "building" in q_lower:
        _append_unique_query(queries, seen, "six building blocks overview AI agents")

    if re.search(r"types?\s+of\s+memory|memory\s+do\s+agents|memory\s+types?", q_lower):
        for subquery in _MEMORY_TYPE_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"design\s+patterns?|agent\s+patterns?", q_lower) or "popular" in q_lower:
        for subquery in _DESIGN_PATTERN_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"sub-?agents?|orchestration", q_lower) and re.search(
        r"roles?", q_lower
    ):
        for subquery in _SUBAGENT_ROLE_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    attention_match = re.search(
        r"(?:pay\s+special\s+attention\s+to|including|covering|focus\s+on)\s+(.+)",
        core,
        re.IGNORECASE,
    )
    if attention_match:
        for topic in _split_topic_clause(attention_match.group(1)):
            _append_unique_query(queries, seen, topic)

    return queries[:max_queries]