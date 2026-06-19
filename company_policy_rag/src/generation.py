"""
Grounded response synthesis with optional faithfulness guard.

Trade-off (production-rag):
  strict mode  → Faithfulness ~1.0, Answer Relevancy lower (over-abstention)
  balanced mode → Faithfulness ≥0.90 target, better helpfulness for internal policy Q&A
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.response_synthesizers.base import QueryTextType
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.code_validation import CodeValidationTrace, apply_code_validation_pipeline
from src.config import settings
from src.language import append_language_hint
from src.prompts import (
    INSUFFICIENT_INFO_MESSAGE,
    LOW_CONFIDENCE_MESSAGE,
    PARTIAL_ANSWER_MIN_CHARS,
    PARTIAL_ANSWER_PREFIX,
    format_nodes_for_prompt,
    get_faithfulness_claim_trim_prompt,
    get_faithfulness_guard_prompt,
    get_refine_template,
    get_text_qa_template,
    resolve_grounding_mode,
)
from src.citations import log_retrieval_stage, record_generation_sources
from src.query_processing import is_comprehensive_list_query
from src.retriever import build_retriever
from src.timing import get_current_timing, record_stage
from src.utils import logger, timer

# Substrings that signal a trailing abstention block (full or partial phrasing).
_ABSTENTION_MARKERS: tuple[str, ...] = (
    INSUFFICIENT_INFO_MESSAGE,
    "The provided documents do not contain sufficient information",
    "The provided document does not contain sufficient information",
)


def _format_text_chunks(nodes: Sequence[NodeWithScore]) -> list[str]:
    """One metadata-rich chunk per retrieved node for compact synthesis."""
    return [format_nodes_for_prompt([n]) for n in nodes]


@dataclass
class GenerationTrace:
    """Full generation trace for eval and debugging."""

    pre_guard_answer: str
    post_guard_answer: str
    final_answer: str
    code_validation: CodeValidationTrace
    fallback_reason: str = "none"


def _finalize_answer(
    query_str: str,
    answer: str,
    nodes: Sequence[NodeWithScore],
    llm: LLM | None,
) -> tuple[str, CodeValidationTrace]:
    guarded = apply_faithfulness_guard(answer, nodes, llm)
    final, code_trace = apply_code_validation_pipeline(query_str, guarded, nodes, llm)
    return final, code_trace


class GroundedCompactAndRefine(CompactAndRefine):
    """
    CompactAndRefine that feeds metadata-tagged sources to the LLM.

    Overrides synthesize so text_chunks use format_node_for_prompt instead of
    raw node text — section titles and page numbers stay visible to the model.
    """

    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        if len(nodes) == 0:
            return self._prepare_response_output(INSUFFICIENT_INFO_MESSAGE, [])

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        text_chunks = _format_text_chunks(nodes)
        query_str = append_language_hint(query.query_str)
        response_str = self.get_response(
            query_str=query_str,
            text_chunks=text_chunks,
            **response_kwargs,
        )
        response_str = normalize_balanced_answer(
            response_str, query=query.query_str
        )
        response_str, _ = _finalize_answer(query_str, response_str, nodes, self._llm)
        additional_source_nodes = additional_source_nodes or []
        source_nodes = list(nodes) + list(additional_source_nodes)
        return self._prepare_response_output(response_str, source_nodes)

    async def asynthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        if len(nodes) == 0:
            return self._prepare_response_output(INSUFFICIENT_INFO_MESSAGE, [])

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        text_chunks = _format_text_chunks(nodes)
        query_str = append_language_hint(query.query_str)
        response_str = await self.aget_response(
            query_str=query_str,
            text_chunks=text_chunks,
            **response_kwargs,
        )
        response_str = normalize_balanced_answer(
            response_str, query=query.query_str
        )
        response_str, _ = _finalize_answer(query_str, response_str, nodes, self._llm)
        additional_source_nodes = additional_source_nodes or []
        source_nodes = list(nodes) + list(additional_source_nodes)
        return self._prepare_response_output(response_str, source_nodes)


def build_grounded_response_synthesizer(llm: LLM | None = None) -> GroundedCompactAndRefine:
    """Build synthesizer with mode-appropriate prompts (strict or balanced)."""
    llm = llm or Settings.llm
    grounded = GroundedCompactAndRefine(
        llm=llm,
        text_qa_template=get_text_qa_template(),
        refine_template=get_refine_template(),
    )
    grounded._empty_response = INSUFFICIENT_INFO_MESSAGE
    return grounded


def _resolve_guard_mode() -> str:
    """Map config to active guard behavior: strict, balanced, or off."""
    if not settings.enable_faithfulness_check:
        return "off"
    if settings.faithfulness_guard_mode == "off":
        return "off"
    if settings.faithfulness_guard_mode == "strict":
        return "strict"
    if settings.faithfulness_guard_mode == "balanced":
        return "balanced"
    return resolve_grounding_mode()


def _has_partial_answer_prefix(text: str) -> bool:
    """True if the answer uses the balanced partial-answer opener (with or without 'in the documents')."""
    lowered = text.strip().lower()
    return (
        PARTIAL_ANSWER_PREFIX in text
        or lowered.startswith("based on the available information")
    )


def _has_substantive_partial_body(text: str) -> bool:
    """True when text looks like a grounded partial answer worth keeping."""
    body = text.strip()
    if len(body) >= PARTIAL_ANSWER_MIN_CHARS:
        return True
    lowered = body.lower()
    return _has_partial_answer_prefix(body) and (
        "[source" in lowered or "excerpts do not" in lowered
    )


def _strip_trailing_abstention_suffix(text: str) -> str:
    """Remove abstention blocks appended after a substantive partial answer."""
    normalized = text.strip()
    if not _has_partial_answer_prefix(normalized):
        return normalized

    for marker in _ABSTENTION_MARKERS:
        if marker not in normalized:
            continue
        before = normalized.split(marker)[0].strip()
        if _has_substantive_partial_body(before):
            logger.debug(
                "Balanced normalize: stripped trailing abstention suffix (kept %d chars)",
                len(before),
            )
            return before
    return normalized


def _strip_leading_abstention_prefix(text: str) -> str:
    """
    Remove abstention-first blocks when a substantive partial answer follows.

    Fixes inverted double-endings (abstention paragraph, then partial answer).
    """
    normalized = text.strip()
    lowered = normalized.lower()
    partial_idx = lowered.find(PARTIAL_ANSWER_PREFIX.lower())
    if partial_idx == -1:
        partial_idx = lowered.find("based on the available information")
    if partial_idx <= 0:
        return normalized

    starts_with_abstention = any(
        normalized.startswith(marker) for marker in _ABSTENTION_MARKERS
    )
    if not starts_with_abstention:
        return normalized

    partial = normalized[partial_idx:].strip()
    if _has_substantive_partial_body(partial):
        logger.debug("Balanced normalize: dropped leading abstention block")
        return _strip_trailing_abstention_suffix(partial)
    return normalized


def _preserve_balanced_partial_answer(answer: str) -> str:
    """
    Strip trailing abstention when a substantive partial answer exists.

    Balanced-mode safety net for the 'partial + full abstention' double-ending pattern.
    """
    return _strip_trailing_abstention_suffix(answer.strip())


def normalize_balanced_answer(answer: str, *, query: str | None = None) -> str:
    """
    Post-process balanced-mode LLM output before the faithfulness guard.

    Strips double-endings and redundant abstention suffixes at synthesis time so
    pre_guard traces and final answers stay consistent.
    """
    if resolve_grounding_mode() != "balanced":
        return answer.strip()

    normalized = answer.strip()
    if not normalized:
        return answer

    normalized = _strip_leading_abstention_prefix(normalized)
    normalized = _strip_trailing_abstention_suffix(normalized)
    if query and is_comprehensive_list_query(query):
        normalized = _strip_trailing_abstention_suffix(normalized)
    return normalized


def _resolve_guard_reject_action() -> str:
    """Return keep|trim|abstain for balanced-guard reject handling."""
    return settings.faithfulness_guard_reject_action


def _trim_unsupported_claims(answer: str, context: str, llm: LLM) -> str:
    """Remove unsupported sentences while preserving cited supported content."""
    prompt = get_faithfulness_claim_trim_prompt().format(
        insufficient_message=INSUFFICIENT_INFO_MESSAGE,
        context=context[:6000],
        answer=answer[:2000],
    )
    trimmed = str(llm.complete(prompt)).strip()
    if not trimmed:
        return answer
    if trimmed.startswith(INSUFFICIENT_INFO_MESSAGE):
        return INSUFFICIENT_INFO_MESSAGE
    if INSUFFICIENT_INFO_MESSAGE in trimmed and len(trimmed) < PARTIAL_ANSWER_MIN_CHARS:
        return INSUFFICIENT_INFO_MESSAGE
    return trimmed


def _parse_guard_verdict(raw: str, guard_mode: str) -> bool:
    """
    Return True if answer passes the guard.

    strict: YES = pass
    balanced: SUPPORTED = pass (or absence of UNSUPPORTED)
    """
    text = raw.strip().upper()
    if guard_mode == "strict":
        return text.startswith("YES")
    # balanced — default pass unless clearly UNSUPPORTED
    if "UNSUPPORTED" in text and "SUPPORTED" not in text.split("UNSUPPORTED")[0]:
        return False
    if text.startswith("UNSUPPORTED"):
        return False
    return True


def apply_faithfulness_guard(
    answer: str,
    nodes: Sequence[NodeWithScore],
    llm: LLM | None = None,
) -> str:
    """
    Second-pass check tuned by FAITHFULNESS_GUARD_MODE.

    strict: reject unless every claim is directly supported (binary YES/NO)
    balanced: reject only clear hallucinations (SUPPORTED/UNSUPPORTED);
              also strips trailing abstention after substantive partial answers
    off: skip guard entirely
    """
    guard_mode = _resolve_guard_mode()
    if guard_mode == "off":
        return answer

    normalized = answer.strip()
    if not normalized:
        return answer

    if guard_mode == "balanced":
        normalized = _preserve_balanced_partial_answer(normalized)

    if normalized == INSUFFICIENT_INFO_MESSAGE:
        return INSUFFICIENT_INFO_MESSAGE

    if not nodes:
        return INSUFFICIENT_INFO_MESSAGE

    judge = llm or Settings.llm
    if judge is None:
        return answer

    context = format_nodes_for_prompt(list(nodes))
    prompt = get_faithfulness_guard_prompt(
        mode="strict" if guard_mode == "strict" else "balanced"
    ).format(
        context=context[:6000],
        answer=normalized[:2000],
    )

    try:
        with timer("faithfulness_guard") as t_guard:
            verdict_raw = str(judge.complete(prompt)).strip()
        if get_current_timing() is not None:
            record_stage("faithfulness_guard", t_guard["elapsed_ms"])
        passed = _parse_guard_verdict(verdict_raw, guard_mode)
        if passed:
            logger.debug("Faithfulness guard (%s): passed", guard_mode)
            return normalized
        logger.warning(
            "Faithfulness guard (%s): rejected (verdict=%s)",
            guard_mode,
            verdict_raw[:30],
        )
        if guard_mode == "balanced":
            action = _resolve_guard_reject_action()
            if action == "abstain":
                return INSUFFICIENT_INFO_MESSAGE
            if action == "trim":
                trimmed = _trim_unsupported_claims(normalized, context, judge)
                if (
                    trimmed
                    and trimmed != INSUFFICIENT_INFO_MESSAGE
                    and len(trimmed.strip()) >= PARTIAL_ANSWER_MIN_CHARS // 2
                ):
                    logger.warning(
                        "Faithfulness guard (balanced): trimmed unsupported claims"
                    )
                    return trimmed
                logger.warning(
                    "Faithfulness guard (balanced): trim produced no usable answer, keeping original"
                )
            else:
                logger.warning(
                    "Faithfulness guard (balanced): keeping original answer to preserve relevancy"
                )
            return normalized
        return INSUFFICIENT_INFO_MESSAGE
    except Exception as exc:
        logger.warning("Faithfulness guard skipped due to error: %s", exc)
        return normalized


def generate_grounded_answer_with_trace(
    query: str,
    nodes: Sequence[NodeWithScore],
    llm: LLM | None = None,
) -> GenerationTrace:
    """
    Run grounded synthesis with full trace (guard + code validation).

    Used by evaluation to detect guard-induced abstention and code fallback regressions.
    """
    empty_trace = CodeValidationTrace()
    if not nodes:
        msg = INSUFFICIENT_INFO_MESSAGE
        return GenerationTrace(
            pre_guard_answer=msg,
            post_guard_answer=msg,
            final_answer=msg,
            code_validation=empty_trace,
            fallback_reason="none",
        )

    synth = build_grounded_response_synthesizer(llm)
    judge_llm = llm or synth._llm
    text_chunks = _format_text_chunks(nodes)
    with timer("generation") as t_gen:
        pre_guard = synth.get_response(query_str=query, text_chunks=text_chunks)
    if get_current_timing() is not None:
        record_stage("generation", t_gen["elapsed_ms"])
    normalized = normalize_balanced_answer(pre_guard, query=query)
    post_guard = apply_faithfulness_guard(normalized, nodes, judge_llm)
    final, code_trace = apply_code_validation_pipeline(query, post_guard, nodes, judge_llm)

    fallback_reason = "none"
    if final == LOW_CONFIDENCE_MESSAGE:
        fallback_reason = "code_validation"
    elif final == INSUFFICIENT_INFO_MESSAGE and post_guard != INSUFFICIENT_INFO_MESSAGE:
        fallback_reason = "faithfulness"

    return GenerationTrace(
        pre_guard_answer=pre_guard,
        post_guard_answer=post_guard,
        final_answer=final,
        code_validation=code_trace,
        fallback_reason=fallback_reason,
    )


class SourceTrackingQueryEngine:
    """
    Proxy around RetrieverQueryEngine that records source_nodes per query.

    UI citations must come from these nodes — the same chunks sent to the LLM —
    not from a separate retriever.retrieve() call.
    """

    def __init__(self, inner: RetrieverQueryEngine) -> None:
        self._inner = inner

    def query(self, str_or_query_bundle: Any) -> Any:
        response = self._inner.query(str_or_query_bundle)
        nodes = list(getattr(response, "source_nodes", None) or [])
        record_generation_sources(nodes)
        log_retrieval_stage("query_engine_output", nodes)
        return response

    async def aquery(self, str_or_query_bundle: Any) -> Any:
        if hasattr(self._inner, "aquery"):
            response = await self._inner.aquery(str_or_query_bundle)
        else:
            response = self._inner.query(str_or_query_bundle)
        nodes = list(getattr(response, "source_nodes", None) or [])
        record_generation_sources(nodes)
        log_retrieval_stage("query_engine_output", nodes)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_grounded_query_engine(
    index: VectorStoreIndex | None = None,
    *,
    filters: dict[str, Any] | None = None,
) -> SourceTrackingQueryEngine:
    """
    Query engine with grounding prompts and optional faithfulness guard.

    Pipeline: rewrite → retrieve → rerank → grounded synthesis → faithfulness check
    Returns a source-tracking wrapper so citations match generation input.
    """
    retriever = build_retriever(index, filters=filters)
    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=build_grounded_response_synthesizer(),
    )
    return SourceTrackingQueryEngine(engine)