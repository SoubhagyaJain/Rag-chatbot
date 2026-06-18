"""
Shared utilities: logging, timing, citation formatting, section detection.

Section detection is one of the highest-ROI metadata improvements for legal/policy
RAG: it enables accurate citations, metadata pre-filtering, and future reranker
signals — without changing the embedding model or vector store.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Literal, TypeVar

from llama_index.core.schema import NodeWithScore, TextNode

from src.config import settings

F = TypeVar("F", bound=Callable[..., Any])

# ── Logging & timing ─────────────────────────────────────────────────────────


def setup_logging(name: str | None = None) -> logging.Logger:
    """Configure module-level logger with file + console handlers."""
    log = logging.getLogger(name or "company_policy_rag")
    if log.handlers:
        return log

    log.setLevel(getattr(logging, settings.log_level))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log.addHandler(console)

    log_file = settings.logs_dir / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    return log


logger = setup_logging()


@contextmanager
def timer(label: str) -> Generator[dict[str, float], None, None]:
    """Context manager that records elapsed milliseconds."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000
        logger.debug("%s completed in %.1f ms", label, result["elapsed_ms"])


def timed(label: str | None = None) -> Callable[[F], F]:
    """Decorator to log function execution time."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = label or func.__name__
            with timer(name) as t:
                out = func(*args, **kwargs)
            logger.info("%s took %.1f ms", name, t["elapsed_ms"])
            return out

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Section detection for policy/legal PDFs ─────────────────────────────────

# Valid Roman numerals commonly used in handbooks (I–XX usually sufficient).
# We use a simple [IVXLC]+ capture + allowlist validation rather than a full
# numeric grammar — handbooks rarely exceed XX and correctness is verified post-match.
_VALID_ROMANS = frozenset(
    {
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    }
)


@dataclass(frozen=True)
class SectionPattern:
    """
    Declarative section-heading pattern.

    Extensibility: append new SectionPattern entries to SECTION_PATTERNS — no
    changes needed elsewhere. Higher priority runs first (lower number = first).
    """

    name: str
    level: int
    regex: re.Pattern[str]
    # Group indices: (number_group, title_group); None means entire match is title
    number_group: int | None
    title_group: int | None
    modes: frozenset[str] = frozenset({"standard", "strict", "permissive"})
    priority: int = 100

    def matches_mode(self, mode: str) -> bool:
        return mode in self.modes


@dataclass
class SectionHeading:
    """Parsed heading from a single line of policy/legal text."""

    level: int
    section_number: str | None
    section_title: str
    full_label: str
    pattern_name: str


@dataclass
class SectionContext:
    """Current hierarchical position while scanning a document."""

    section_title: str | None = None
    section_number: str | None = None
    section_path: str | None = None
    section_level: int | None = None
    headings: list[SectionHeading] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "section_title": self.section_title,
            "section_number": self.section_number,
            "section_path": self.section_path,
            "section_level": self.section_level,
        }


def _build_section_patterns() -> list[SectionPattern]:
    """
    Regex patterns tuned for Employee Handbooks, HR policies, and legal docs.

    Level semantics (handbook convention):
      1 = top-level parts   (Roman: I., II.)
      2 = sub-parts         (Letter: A., B.)
      3 = numbered sections (5.2, 1.2.3)
      4 = formal legal refs  (Article 7, Section 3.1, Clause 2)
      5 = stylistic headings (ALL CAPS lines)
    """
    return sorted(
        [
            SectionPattern(
                name="article_section_clause",
                level=4,
                regex=re.compile(
                    r"^(?:Article|Section|Clause|Part|Chapter|Appendix)\s+"
                    r"([\dIVXLC]+(?:\.\d+)*)\s*"
                    r"(?::|\.|\-|–|—)\s*(.+)$",
                    re.IGNORECASE,
                ),
                number_group=1,
                title_group=2,
                priority=10,
            ),
            SectionPattern(
                name="roman_numeral",
                level=1,
                # Title may be ALL CAPS in handbooks — matched separately from all_caps pattern
                regex=re.compile(r"^([IVXLC]+)\.\s+(.+)$", re.IGNORECASE),
                number_group=1,
                title_group=2,
                priority=20,
            ),
            SectionPattern(
                name="letter_subsection",
                level=2,
                regex=re.compile(r"^([A-Z])\.\s+([A-Z][A-Za-z0-9\s\-,&'()/]{2,})$"),
                number_group=1,
                title_group=2,
                modes=frozenset({"standard", "permissive"}),
                priority=30,
            ),
            SectionPattern(
                name="numbered_section",
                level=3,
                regex=re.compile(
                    r"^(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z0-9\s\-,&'()/]{3,})$"
                ),
                number_group=1,
                title_group=2,
                priority=40,
            ),
            SectionPattern(
                name="numbered_trailing_dot",
                level=3,
                regex=re.compile(
                    r"^(\d+(?:\.\d+){0,3})\.\s+([A-Z][A-Za-z0-9\s\-,&'()/]{3,})$"
                ),
                number_group=1,
                title_group=2,
                priority=45,
            ),
            SectionPattern(
                name="all_caps_heading",
                level=5,
                regex=re.compile(r"^([A-Z][A-Z0-9\s\-/&]{5,})$"),
                number_group=None,
                title_group=1,
                modes=frozenset({"standard", "permissive"}),
                priority=60,
            ),
        ],
        key=lambda p: p.priority,
    )


SECTION_PATTERNS: list[SectionPattern] = _build_section_patterns()


def _is_noise_line(line: str) -> bool:
    """Filter lines that look like headings but are page furniture or boilerplate."""
    lower = line.lower().strip()
    noise_prefixes = (
        "page ",
        "confidential and proprietary",
        "strictly confidential",
        "proprietary and confidential",
        "table of contents",
        "revised ",
        "effective date",
        "last updated",
    )
    if any(lower.startswith(p) for p in noise_prefixes):
        return True
    # Mostly digits/punctuation — page numbers, dates
    alpha = sum(c.isalpha() for c in line)
    return alpha < max(3, len(line) * 0.3)


def _is_valid_roman(number: str) -> bool:
    return number.upper() in _VALID_ROMANS


def _clean_title(title: str) -> str:
    """Normalize extracted title text."""
    cleaned = re.sub(r"\s+", " ", title.strip())
    # Strip trailing punctuation artifacts from PDF extraction
    return cleaned.rstrip(".:;-–—")[:200]


def _heading_from_match(
    pattern: SectionPattern,
    match: re.Match[str],
    *,
    original_line: str,
) -> SectionHeading | None:
    if pattern.number_group is not None:
        number = match.group(pattern.number_group).strip()
        title = _clean_title(match.group(pattern.title_group))
    else:
        number = ""
        title = _clean_title(match.group(pattern.title_group))

    if not title or _is_noise_line(title):
        return None

    if pattern.name == "roman_numeral" and not _is_valid_roman(number):
        return None

    if pattern.name == "all_caps_heading":
        # Require majority uppercase to avoid matching regular sentences
        upper = sum(c.isupper() for c in title if c.isalpha())
        alpha = sum(c.isalpha() for c in title)
        if alpha == 0 or upper / alpha < 0.7:
            return None
        number = ""

    if pattern.name == "letter_subsection":
        # Single-letter subsections almost always use A–Z only
        if len(number) != 1:
            return None

    # Preserve the original line for citations — avoids "5.2." double-dot artifacts
    full_label = original_line.strip()[:200]
    return SectionHeading(
        level=pattern.level,
        section_number=number or None,
        section_title=title,
        full_label=full_label,
        pattern_name=pattern.name,
    )


def parse_section_heading(
    line: str,
    *,
    mode: str | None = None,
) -> SectionHeading | None:
    """
    Parse a single line into a SectionHeading, or return None.

    Production rationale: line-by-line regex is fast, deterministic, and easy to
    unit-test — preferable to LLM-based structure extraction for v1 indexing.
    """
    if not settings.enable_section_detection:
        return None

    stripped = line.strip()
    if not stripped or len(stripped) < 3 or _is_noise_line(stripped):
        return None

    detection_mode = mode or settings.section_detection_mode
    for pattern in SECTION_PATTERNS:
        if not pattern.matches_mode(detection_mode):
            continue
        match = pattern.regex.match(stripped)
        if match:
            heading = _heading_from_match(pattern, match, original_line=stripped)
            if heading:
                return heading
    return None


def scan_text_for_headings(
    text: str,
    *,
    max_lines: int | None = None,
    mode: str | None = None,
) -> list[SectionHeading]:
    """Return all headings found in text (top-to-bottom order)."""
    headings: list[SectionHeading] = []
    lines = text.splitlines()
    if max_lines is not None:
        lines = lines[:max_lines]

    for line in lines:
        heading = parse_section_heading(line, mode=mode)
        if heading:
            headings.append(heading)
    return headings


def detect_section_title(text: str) -> str | None:
    """
    Backward-compatible helper: return the first detected heading label in text.

    Prefer SectionTracker + enrich_* functions for indexing pipelines.
    """
    headings = scan_text_for_headings(text, max_lines=20)
    if headings:
        return headings[0].full_label
    return None


class SectionTracker:
    """
    Maintains a hierarchical section stack while scanning documents in reading order.

    Critical production behavior: when a new top-level Roman section appears,
    all deeper levels (letters, numbered subsections) are cleared — mirroring
    how handbooks are structured and preventing stale subsection paths.
    """

    def __init__(self) -> None:
        self._stack: dict[int, SectionHeading] = {}

    def reset(self) -> None:
        self._stack.clear()

    def update(self, heading: SectionHeading | None) -> SectionContext:
        """Push heading onto stack, clearing deeper levels."""
        if heading is None:
            return self.current_context()
        self._stack = {lvl: h for lvl, h in self._stack.items() if lvl < heading.level}
        self._stack[heading.level] = heading
        return self.current_context()

    def update_from_text(self, text: str) -> SectionContext:
        """Scan all lines in text and apply every heading found (in order)."""
        for line in text.splitlines():
            heading = parse_section_heading(line)
            if heading:
                self.update(heading)
        return self.current_context()

    def current_context(self) -> SectionContext:
        if not self._stack:
            return SectionContext()

        deepest_level = max(self._stack)
        deepest = self._stack[deepest_level]
        path = " > ".join(
            self._stack[lvl].full_label for lvl in sorted(self._stack)
        )
        return SectionContext(
            section_title=deepest.section_title,
            section_number=deepest.section_number,
            section_path=path,
            section_level=deepest.level,
            headings=list(self._stack[lvl] for lvl in sorted(self._stack)),
        )


def enrich_text_with_section_context(
    text: str,
    tracker: SectionTracker,
    *,
    scan_max_lines: int | None = None,
) -> SectionContext:
    """
    Apply tracker to a text block: propagate prior context, then scan for new headings.

    scan_max_lines=None scans the full text (used for chunks); page headers use
    a higher limit only at document level in indexing.py.
    """
    context = tracker.current_context()
    lines = text.splitlines()
    if scan_max_lines is not None:
        lines = lines[:scan_max_lines]

    for line in lines:
        heading = parse_section_heading(line)
        if heading:
            context = tracker.update(heading)
    return context


def section_metadata_from_context(context: SectionContext) -> dict[str, Any]:
    """Convert SectionContext to flat metadata dict for LlamaIndex nodes."""
    meta = context.to_metadata()
    # Ensure JSON-serializable values for vector store persistence
    return {k: v for k, v in meta.items() if v is not None}


# ── Category & citation formatting ───────────────────────────────────────────


def infer_category(file_path: Path, document_type: str) -> str:
    """Derive a human-readable category from filename and folder."""
    stem = file_path.stem.replace("_", " ").replace("-", " ").title()
    if document_type == "legal_document":
        return f"Legal — {stem}"
    if document_type == "company_policy":
        return f"Policy — {stem}"
    return stem


def format_citation(node: TextNode | NodeWithScore) -> dict[str, Any]:
    """
    Normalize chunk metadata into a citation dict for the UI and agent.

    Rich citations are critical in policy/legal RAG: users must verify claims
    against the exact source page, section path, and clause number.
    """
    from src.pdf_images import get_page_images

    base = node.node if isinstance(node, NodeWithScore) else node
    meta = base.metadata or {}
    score = node.score if isinstance(node, NodeWithScore) else None

    source_file = meta.get("source_file", "unknown")
    page_number = meta.get("page_number")
    page_images = get_page_images(source_file, page_number)
    max_images = settings.citation_max_page_images

    return {
        "source_file": source_file,
        "page_number": page_number,
        "section_title": meta.get("section_title"),
        "section_number": meta.get("section_number"),
        "section_path": meta.get("section_path"),
        "document_type": meta.get("document_type"),
        "category": meta.get("category"),
        "file_path": meta.get("file_path"),
        "score": round(score, 4) if score is not None else None,
        "excerpt": (base.text or "")[:300],
        "page_images": [str(path) for path in page_images[:max_images]],
    }


def format_citations(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """Format a list of retrieved nodes as citations."""
    return [format_citation(n) for n in nodes]


def shorten_source_filename(filename: str) -> str:
    """
    Convert 'employee_handbook.pdf' → 'Employee Handbook' for readable citations.

    Keeps the original extension-free stem humanized; users recognize document
    names faster than raw filesystem paths.
    """
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").strip().title() or filename


def _page_suffix(page_number: int | None, *, long_form: bool = False) -> str:
    if page_number is None:
        return ""
    return f"(Page {page_number})" if long_form else f"(p.{page_number})"


def format_citation_primary(
    citation: dict[str, Any],
    *,
    fmt: str | None = None,
) -> str:
    """
    Single-line citation label for UI display.

    section_first (default):
        II. GENERAL EMPLOYMENT INFORMATION > 5.2 Vacation Benefits (p.14)

    document_first:
        Employee Handbook.pdf — Section 5.2 Vacation Benefits (Page 14)
    """
    style = fmt or settings.citation_format
    source = citation.get("source_file", "unknown")
    short_name = shorten_source_filename(source)
    page = citation.get("page_number")
    section_path = citation.get("section_path")
    section_title = citation.get("section_title")
    section_number = citation.get("section_number")
    page_str = _page_suffix(page, long_form=(style == "document_first"))

    if style == "document_first":
        section_label = section_path or _section_label(section_title, section_number)
        parts = [short_name]
        if section_label:
            parts.append(f"— {section_label}")
        if page_str:
            parts.append(page_str)
        return " ".join(parts)

    # section_first — prioritize structural path for legal verification
    if section_path:
        return f"{section_path} {page_str}".strip()
    section_label = _section_label(section_title, section_number)
    if section_label:
        return f"{section_label} {page_str}".strip()
    if page_str:
        return f"{short_name} {page_str}".strip()
    return short_name


def _section_label(
    section_title: str | None,
    section_number: str | None,
) -> str | None:
    if section_title and section_number:
        return f"Section {section_number} {section_title}"
    if section_title:
        return section_title
    if section_number:
        return f"Section {section_number}"
    return None


def _citation_dedupe_key(citation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        citation.get("source_file"),
        citation.get("page_number"),
        citation.get("section_path") or citation.get("section_title"),
    )


def dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collapse duplicate sources (same file + page + section).

    When multiple chunks hit the same policy clause, show one citation — keeps
    the UI clean while preserving the highest-relevance excerpt.
    """
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for citation in citations:
        key = _citation_dedupe_key(citation)
        existing = seen.get(key)
        if existing is None:
            seen[key] = citation
            continue
        # Keep the chunk with the better retrieval score
        if (citation.get("score") or 0) > (existing.get("score") or 0):
            seen[key] = citation
    return list(seen.values())


def prepare_citations_for_display(
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort, dedupe, and cap citations for the Chainlit UI.

    Centralizes display policy so chat_app.py stays thin and testable.
    """
    if not citations:
        return []

    sorted_citations = sorted(
        citations,
        key=lambda c: c.get("score") or 0,
        reverse=True,
    )

    if settings.citation_dedupe:
        sorted_citations = dedupe_citations(sorted_citations)

    return sorted_citations[: settings.citation_max_sources]


def format_citation_excerpt(citation: dict[str, Any], max_len: int = 280) -> str:
    """Trim excerpt for expandable source preview; ellipsis if truncated."""
    excerpt = (citation.get("excerpt") or "").strip()
    if not excerpt:
        return "_No preview available for this source._"
    excerpt = re.sub(r"\s+", " ", excerpt)
    if len(excerpt) <= max_len:
        return f'"{excerpt}"'
    return f'"{excerpt[:max_len].rstrip()}…"'


def citations_to_markdown(citations: list[dict[str, Any]]) -> str:
    """
    Render citations as markdown (fallback / logging / non-Chainlit contexts).

    Prefer build_chainlit_citation_elements() in the chat app for rich UI.
    """
    prepared = prepare_citations_for_display(citations)
    if not prepared:
        return "_No sources retrieved._"

    lines = [f"### Sources ({len(prepared)})"]
    for i, c in enumerate(prepared, 1):
        label = format_citation_primary(c)
        lines.append(f"{i}. **{label}**")
        if c.get("selection_reason"):
            lines.append(f"   - Selected: {c['selection_reason']}")
        if settings.citation_show_relevance_score and c.get("score") is not None:
            lines.append(f"   - Relevance: {c['score']}")
        if settings.citation_show_excerpts and c.get("excerpt"):
            lines.append(f"   > {format_citation_excerpt(c)}")
    return "\n".join(lines)


def build_chainlit_citation_elements(citations: list[dict[str, Any]]) -> list[Any]:
    """
    Build Chainlit Text elements — expandable source cards below the answer.

    Design: each source is a collapsible element (click to expand excerpt).
    This keeps the main answer clean while letting users verify policy text.
    """
    import chainlit as cl

    prepared = prepare_citations_for_display(citations)
    elements: list[Any] = []

    for i, citation in enumerate(prepared, 1):
        label = format_citation_primary(citation)
        if settings.citation_show_relevance_score and citation.get("score") is not None:
            label = f"{label} · score {citation['score']:.2f}"

        content_parts: list[str] = []
        if citation.get("selection_reason"):
            reason = citation["selection_reason"]
            if reason == "cited_in_answer":
                content_parts.append("**Selected because:** cited in the answer")
            else:
                content_parts.append("**Selected because:** high relevance score")
        source = citation.get("source_file", "unknown")
        content_parts.append(f"**Document:** {shorten_source_filename(source)}")
        if citation.get("section_number"):
            content_parts.append(f"**Section #:** {citation['section_number']}")
        if citation.get("section_path") and citation.get("section_path") != label:
            content_parts.append(f"**Full path:** {citation['section_path']}")
        if citation.get("page_number") is not None:
            content_parts.append(f"**Page:** {citation['page_number']}")
        if settings.citation_show_excerpts:
            content_parts.append(f"\n**Excerpt:** {format_citation_excerpt(citation)}")

        elements.append(
            cl.Text(
                name=f"{i}. {label}",
                content="\n".join(content_parts),
                display="inline",
            )
        )

    return elements