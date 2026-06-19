"""
Post-generation code-line validation and self-correction (Phase 3).

Runs after the general faithfulness guard when answers contain code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore

from src.config import settings
from src.prompts import (
    LOW_CONFIDENCE_MESSAGE,
    format_nodes_for_prompt,
    get_code_self_correction_prompt,
    get_code_validation_prompt,
)
from src.timing import get_current_timing, record_stage
from src.utils import logger, timer

_FENCE_RE = re.compile(r"```[\w-]*\n(.*?)```", re.DOTALL)
_CODE_BLOCK_PREFIX = "[CODE BLOCK"
_CODE_BLOCK_HEADER_RE = re.compile(r"\[CODE BLOCK[^\]]*\]\s*", re.IGNORECASE)
_SOURCE_TAG_RE = re.compile(r"<source[^>]*>|</source>", re.IGNORECASE)
_DEF_NAME_RE = re.compile(r"^(?:async\s+)?def\s+(\w+)")
_ORPHAN_CODE_PREAMBLE_RE = re.compile(
    r"(?im)^.*\b(?:defined|implemented|shown)\s+as\s*:\s*$"
)
_CONTRADICTORY_ABSENCE_RE = re.compile(
    r"(?i)(?:does not provide|do not contain|cannot find|no (?:complete )?code)",
)


def _escape_format_braces(text: str) -> str:
    """Escape braces so str.format on prompt templates does not treat code as placeholders."""
    return text.replace("{", "{{").replace("}", "}}")


@dataclass
class HeuristicResult:
    passed: bool | None
    failed: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class CodeValidationTrace:
    triggered: bool = False
    passed: bool | None = None
    self_corrected: bool = False
    fallback_applied: bool = False
    explanation: str = ""
    validation_method: str = ""
    failed_lines: list[str] = field(default_factory=list)
    matched_lines: list[str] = field(default_factory=list)


def answer_contains_code(text: str) -> bool:
    if not text:
        return False
    if _CODE_BLOCK_PREFIX in text:
        return True
    if "```" in text:
        return True
    return bool(re.search(r"^\s*(def |class |import |from |async def )", text, re.MULTILINE))


def context_contains_code(nodes: Sequence[NodeWithScore]) -> bool:
    for nws in nodes:
        meta = nws.metadata or {}
        if meta.get("content_type") == "code":
            return True
        text = nws.get_content() or ""
        if _CODE_BLOCK_PREFIX in text or "```" in text:
            return True
    return False


def should_validate_code(answer: str, nodes: Sequence[NodeWithScore]) -> bool:
    if not settings.enable_code_validation:
        return False
    if answer_contains_code(answer):
        return True
    if settings.code_validation_trigger_mode == "answer_or_context":
        return context_contains_code(nodes)
    return False


def extract_code_lines(text: str) -> list[str]:
    """Extract normalized non-empty code lines from fenced blocks or code-like lines."""
    lines: list[str] = []
    for match in _FENCE_RE.finditer(text):
        block = match.group(1)
        for line in block.splitlines():
            normalized = _normalize_code_line(line)
            if normalized:
                lines.append(normalized)

    if not lines:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("[Source"):
                continue
            if re.match(r"^(def |class |import |from |async def |#)", stripped):
                normalized = _normalize_code_line(stripped)
                if normalized:
                    lines.append(normalized)
    return lines


def _normalize_code_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return re.sub(r"\s+", " ", stripped).lower()


def _normalize_context_for_matching(context: str) -> str:
    text = _SOURCE_TAG_RE.sub(" ", context)
    text = _CODE_BLOCK_HEADER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).lower()


def _line_matches_context(line: str, context_normalized: str) -> bool:
    if line in context_normalized:
        return True
    def_match = _DEF_NAME_RE.match(line)
    if def_match:
        name = def_match.group(1)
        if f"def {name}" in context_normalized or f"async def {name}" in context_normalized:
            return True
    return False


def heuristic_code_grounded(answer: str, context: str) -> HeuristicResult:
    """
    Fast check: code lines in answer appear in context.

    Returns passed=None when there are no extractable code lines.
    """
    code_lines = extract_code_lines(answer)
    if not code_lines:
        return HeuristicResult(None)

    context_normalized = _normalize_context_for_matching(context)
    matched: list[str] = []
    failed: list[str] = []

    for line in code_lines:
        if _line_matches_context(line, context_normalized):
            matched.append(line)
        else:
            failed.append(line)

    if not failed:
        return HeuristicResult(
            True,
            matched=matched,
            explanation="heuristic: all code lines found in context",
        )

    ratio = len(matched) / len(code_lines)
    min_ratio = settings.code_validation_heuristic_min_ratio
    if ratio >= min_ratio:
        return HeuristicResult(
            True,
            matched=matched,
            failed=failed,
            explanation=f"heuristic: {len(matched)}/{len(code_lines)} lines matched (ratio {ratio:.2f})",
        )

    return HeuristicResult(
        False,
        matched=matched,
        failed=failed,
        explanation=f"heuristic: {len(failed)} line(s) not in context",
    )


def remove_orphan_code_preambles(answer: str) -> str:
    """Drop lines like 'the tool is defined as:' when no code block follows."""
    if not answer.strip():
        return answer

    lines = answer.splitlines()
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        if _ORPHAN_CODE_PREAMBLE_RE.match(line.strip()):
            remainder = "\n".join(lines[i + 1 :]).strip()
            if not remainder or "```" not in remainder:
                continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


def _needs_prose_fallback(answer: str) -> bool:
    """True when stripped answer still looks broken or self-contradictory."""
    text = answer.strip()
    if not text:
        return True
    if _ORPHAN_CODE_PREAMBLE_RE.search(text):
        return True
    has_substance = len(text) >= 80 and "[Source" in text
    has_denial = bool(_CONTRADICTORY_ABSENCE_RE.search(text))
    return has_denial and has_substance


def strip_unsupported_code(answer: str, failed_lines: list[str]) -> str:
    """Remove fenced blocks with unsupported code; keep surrounding prose."""
    if not answer.strip():
        return answer

    failed_set = set(failed_lines)
    parts: list[str] = []
    last_end = 0

    for match in _FENCE_RE.finditer(answer):
        parts.append(answer[last_end : match.start()])
        block_body = match.group(1)
        block_lines = [ln for ln in extract_code_lines(f"```\n{block_body}\n```")]
        block_failed = any(ln in failed_set for ln in block_lines) if failed_set else True
        if not block_failed:
            parts.append(match.group(0))
        last_end = match.end()

    parts.append(answer[last_end:])
    result = "\n".join(p.strip() for p in parts if p.strip())
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def _parse_validation_verdict(raw: str) -> tuple[bool, str]:
    text = raw.strip()
    upper = text.upper()
    explanation = ""
    for line in text.splitlines():
        if line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[-1].strip()

    if "VERDICT: YES" in upper or upper.startswith("YES"):
        return True, explanation
    if "VERDICT: NO" in upper or upper.startswith("NO"):
        return False, explanation
    if "YES" in upper.split()[:3]:
        return True, explanation
    if "NO" in upper.split()[:3]:
        return False, explanation
    return False, text[:200]


def validate_code_grounding(
    answer: str,
    nodes: Sequence[NodeWithScore],
    llm: LLM,
) -> tuple[bool, str, str, list[str], list[str]]:
    """
    Validate code lines; uses heuristic first when enabled.

    Returns (passed, explanation, method, failed_lines, matched_lines).
    """
    context = format_nodes_for_prompt(list(nodes))

    if settings.code_validation_use_heuristic:
        result = heuristic_code_grounded(answer, context)
        if result.passed is True:
            return (
                True,
                result.explanation,
                "heuristic",
                result.failed,
                result.matched,
            )
        if result.passed is False:
            failed_lines = result.failed
            matched_lines = result.matched
        else:
            failed_lines = []
            matched_lines = []
    else:
        failed_lines = []
        matched_lines = []

    prompt_template = get_code_validation_prompt(
        mode=settings.code_validation_judge_mode,
        failed_lines=failed_lines or None,
    )
    prompt = prompt_template.format(
        context=_escape_format_braces(context[:6000]),
        answer=_escape_format_braces(answer[:2000]),
    )
    try:
        raw = str(llm.complete(prompt)).strip()
        passed, explanation = _parse_validation_verdict(raw)
        return passed, explanation, "llm", failed_lines, matched_lines
    except Exception as exc:
        logger.warning("Code validation judge failed: %s", exc)
        if settings.code_validation_use_heuristic:
            result = heuristic_code_grounded(answer, context)
            if result.passed is not None:
                return (
                    result.passed,
                    "heuristic fallback after judge error",
                    "heuristic",
                    result.failed,
                    result.matched,
                )
        return True, "skipped due to judge error", "skipped", [], []


def self_correct_code_answer(
    query: str,
    answer: str,
    nodes: Sequence[NodeWithScore],
    llm: LLM,
) -> str:
    context = format_nodes_for_prompt(list(nodes))
    prompt = get_code_self_correction_prompt().format(
        query=_escape_format_braces(query[:500]),
        context=_escape_format_braces(context[:6000]),
        answer=_escape_format_braces(answer[:2000]),
    )
    try:
        corrected = str(llm.complete(prompt)).strip()
        return corrected if corrected else answer
    except Exception as exc:
        logger.warning("Code self-correction failed: %s", exc)
        return answer


def _apply_fail_mode(
    query: str,
    answer: str,
    corrected: str,
    nodes: Sequence[NodeWithScore],
    trace: CodeValidationTrace,
    llm: LLM | None = None,
) -> tuple[str, CodeValidationTrace]:
    if settings.code_validation_fail_mode == "strip_code":
        stripped = strip_unsupported_code(corrected, trace.failed_lines)
        stripped = remove_orphan_code_preambles(stripped)

        if stripped.strip() and stripped != LOW_CONFIDENCE_MESSAGE:
            if _needs_prose_fallback(stripped) and llm is not None:
                if settings.enable_code_self_correction:
                    with timer("code_self_correction") as t_fix:
                        stripped = self_correct_code_answer(query, stripped, nodes, llm)
                    if get_current_timing() is not None:
                        record_stage("code_self_correction", t_fix["elapsed_ms"])
                    stripped = remove_orphan_code_preambles(stripped)
                    trace.self_corrected = True

            if stripped.strip() and stripped != LOW_CONFIDENCE_MESSAGE and not _needs_prose_fallback(stripped):
                trace.passed = True
                trace.fallback_applied = False
                trace.validation_method = "strip_code"
                trace.explanation = (
                    f"{trace.explanation}; stripped unsupported code, kept prose"
                ).strip("; ")
                logger.info("Code validation: stripped unsupported code, kept prose")
                return stripped, trace

    trace.fallback_applied = True
    logger.warning("Code validation failed after self-correction — low-confidence fallback")
    return LOW_CONFIDENCE_MESSAGE, trace


def apply_code_validation_pipeline(
    query: str,
    answer: str,
    nodes: Sequence[NodeWithScore],
    llm: LLM | None,
) -> tuple[str, CodeValidationTrace]:
    """
    Validate code grounding; self-correct once on failure; strip or fallback if still failing.
    """
    trace = CodeValidationTrace()
    normalized = answer.strip()
    if not normalized or not nodes:
        return answer, trace

    if not should_validate_code(normalized, nodes):
        return normalized, trace

    if not extract_code_lines(normalized):
        trace.triggered = True
        trace.passed = True
        trace.validation_method = "no_code"
        trace.explanation = "no code lines in answer"
        return normalized, trace

    trace.triggered = True
    judge = llm
    if judge is None:
        return normalized, trace

    with timer("code_validation") as t_val:
        passed, explanation, method, failed, matched = validate_code_grounding(
            normalized, nodes, judge
        )
    if get_current_timing() is not None:
        record_stage("code_validation", t_val["elapsed_ms"])

    trace.passed = passed
    trace.explanation = explanation
    trace.validation_method = method
    trace.failed_lines = failed
    trace.matched_lines = matched

    if passed:
        logger.debug("Code validation passed (%s)", method)
        return normalized, trace

    logger.warning(
        "Code validation failed (%s): %s | failed_lines=%s",
        method,
        explanation[:120],
        failed[:3],
    )

    corrected = normalized
    if settings.enable_code_self_correction and settings.code_self_correction_max_retries > 0:
        with timer("code_self_correction") as t_fix:
            corrected = self_correct_code_answer(query, normalized, nodes, judge)
        if get_current_timing() is not None:
            record_stage("code_self_correction", t_fix["elapsed_ms"])
        trace.self_corrected = True

        with timer("code_validation") as t_reval:
            passed, explanation, method, failed, matched = validate_code_grounding(
                corrected, nodes, judge
            )
        trace.passed = passed
        trace.explanation = explanation
        trace.validation_method = method
        trace.failed_lines = failed
        trace.matched_lines = matched
        if passed:
            logger.info("Code self-correction succeeded")
            return corrected, trace

    return _apply_fail_mode(query, normalized, corrected, nodes, trace, llm=judge)