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
    (
        ("currency", "convert_currency", "exchange rate", "real-world capability", "conversion tool"),
        "convert_currency real-time currency conversion exchange rate tool invocation",
    ),
    (
        ("custom tool", "build custom", "create tool", "tool for an agent"),
        "custom tools MCP function implementation agent tools building block",
    ),
    (
        ("code is available", "full code", "code examples", "where does the guidebook point"),
        "code is available Check this code dailydoseofds link repository",
    ),
    (
        ("check this out", "code walkthrough", "walkthrough"),
        "Check this out code walkthrough example snippet",
    ),
    (
        ("guardrails", "guardrail"),
        "Guardrails building block safety constraints limits validation checkpoints",
    ),
    (
        ("planning building block",),
        "Planning building block six building blocks 5 Levels subdividing tasks",
    ),
    (
        ("manager agent", "multi-agent setup"),
        "manager agent coordinates sub-agents multi-agent pattern",
    ),
    (
        ("rag", "agent workflow"),
        "Agentic RAG retriever agent workflow vector DB context",
    ),
    (
        ("memory work", "memory as a building"),
        "Memory short-term long-term entity memory building block",
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
    re.compile(r"\bguardrails?\b", re.IGNORECASE),
    re.compile(r"\bplanning\s+building\s+block\b", re.IGNORECASE),
    re.compile(r"\bmanager\s+agent\b", re.IGNORECASE),
    re.compile(r"\brag\b.+\bagent\b", re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+memory\b", re.IGNORECASE),
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

_CURRENCY_TOOL_SUBQUERIES: tuple[str, ...] = (
    "convert_currency real-time currency conversion tool",
    "currency conversion tool example invocation exchange rate",
    "real-world capability currency tool demonstrate",
)

_CODE_LINKS_SUBQUERIES: tuple[str, ...] = (
    "code is available full code examples guidebook link",
    "Check this code dailydoseofds",
)

_CHECK_THIS_OUT_SUBQUERIES: tuple[str, ...] = (
    "Check this out code walkthrough",
    "Check this out currency conversion",
    "Check this out custom tool MCP",
)

_CODE_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcurrency\b", re.IGNORECASE),
    re.compile(r"\bconvert_currency\b", re.IGNORECASE),
    re.compile(r"\breal-?world\b.+\btool\b", re.IGNORECASE),
    re.compile(r"\bcustom\s+tool", re.IGNORECASE),
    re.compile(r"\bcode\s+(is\s+)?available\b", re.IGNORECASE),
    re.compile(r"\bcheck\s+this\s+out\b", re.IGNORECASE),
    re.compile(r"\bcode\s+walkthrough", re.IGNORECASE),
    re.compile(r"\bfull\s+code\s+examples?\b", re.IGNORECASE),
    re.compile(r"\bcode\s+examples?\b", re.IGNORECASE),
    re.compile(r"\bshow\b.+\btool\s+example", re.IGNORECASE),
    re.compile(r"\bwhere\b.+\bpoint\b.+\bcode\b", re.IGNORECASE),
    re.compile(r"\bconversion\s+tool\b", re.IGNORECASE),
)


_GUIDEBOOK_EDGE_CASE_MARKERS = (
    "vacation",
    "pto",
    "sick leave",
    "nonprofit",
    "leave policy",
    "employee benefits",
    "health benefits",
)


def is_guidebook_edge_case_query(query: str) -> bool:
    """True when the question asks about HR/policy topics absent from the AI agents guidebook."""
    q = query.lower()
    return any(marker in q for marker in _GUIDEBOOK_EDGE_CASE_MARKERS)


def is_code_or_tool_query(query: str) -> bool:
    """True when the user asks about code examples, tools, or currency walkthroughs."""
    text = query.strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _CODE_TOOL_PATTERNS)


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

    if re.search(r"guardrails?", q_lower):
        _append_unique_query(
            queries,
            seen,
            "Guardrails building block AI agents why used safety constraints",
        )
        _append_unique_query(queries, seen, "Examples of useful guardrails agents")

    if re.search(r"building\s+blocks?", q_lower):
        _append_unique_query(queries, seen, "six building blocks overview AI agents")
        for subquery in _BUILDING_BLOCK_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"planning\s+building\s+block", q_lower):
        _append_unique_query(queries, seen, "Planning building block AI agents")

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

    if re.search(r"currency|convert_currency|exchange\s+rate|conversion\s+tool", q_lower):
        for subquery in _CURRENCY_TOOL_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(
        r"real-?world\b.+\btool\b|currency\s+tool",
        core,
        re.IGNORECASE,
    ):
        for subquery in _CURRENCY_TOOL_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"code\s+(is\s+)?available|full\s+code|code\s+example", q_lower):
        for subquery in _CODE_LINKS_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"check\s+this\s+out|code\s+walkthrough", q_lower):
        for subquery in _CHECK_THIS_OUT_SUBQUERIES:
            _append_unique_query(queries, seen, subquery)

    if re.search(r"custom\s+tool|build\s+custom", q_lower):
        _append_unique_query(queries, seen, "custom tools MCP function implementation")
        _append_unique_query(queries, seen, "custom tools via MCP @mcp.tool CrewAI")

    if re.search(r"manager\s+agent", q_lower):
        _append_unique_query(
            queries,
            seen,
            "manager agent coordinates sub-agents multi-agent pattern",
        )

    if re.search(r"rag", q_lower) and re.search(r"agent", q_lower):
        _append_unique_query(
            queries,
            seen,
            "Agentic RAG retriever agent workflow vector DB",
        )

    if re.search(r"memory", q_lower) and re.search(
        r"building\s+block|how\s+does\s+memory", q_lower
    ):
        _append_unique_query(
            queries,
            seen,
            "Memory short-term long-term entity memory agents improve",
        )

    attention_match = re.search(
        r"(?:pay\s+special\s+attention\s+to|including|covering|focus\s+on)\s+(.+)",
        core,
        re.IGNORECASE,
    )
    if attention_match:
        for topic in _split_topic_clause(attention_match.group(1)):
            _append_unique_query(queries, seen, topic)

    return queries[:max_queries]