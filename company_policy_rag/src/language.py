"""Response language detection and prompt instructions."""

from __future__ import annotations

from typing import Literal

from src.config import settings

ResponseLanguage = Literal["english", "chinese", "match_query"]


def detect_query_language(query: str) -> Literal["english", "chinese"]:
    """Heuristic language detection from the user question."""
    text = query.strip()
    if not text:
        return "english"

    cjk = sum(
        1
        for char in text
        if "\u4e00" <= char <= "\u9fff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
    )
    latin = sum(1 for char in text if char.isascii() and char.isalpha())

    if cjk >= 3 and cjk >= latin:
        return "chinese"
    return "english"


def resolve_response_language(query: str) -> Literal["english", "chinese"]:
    """Return the language the assistant must write in."""
    configured = settings.response_language
    if configured == "english":
        return "english"
    if configured == "chinese":
        return "chinese"
    return detect_query_language(query)


def language_instruction_for_query(query: str) -> str:
    """Short instruction appended to generation/agent prompts."""
    lang = resolve_response_language(query)
    if lang == "chinese":
        return (
            "LANGUAGE: Write the entire answer in Chinese. "
            "Use [Source N] citation tags exactly as shown in the excerpts."
        )
    return (
        "LANGUAGE: Write the entire answer in English only (no Chinese or other languages). "
        "Use [Source N] citation tags exactly — never 来源, 来源1, or translated tag names."
    )


def append_language_hint(query: str) -> str:
    """Append a language hint to the user query for the LLM."""
    hint = language_instruction_for_query(query)
    return f"{query.strip()}\n\n[{hint}]"