"""Tests for grounding prompts and context formatting."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.prompts import (
    AGENT_SYSTEM_PROMPT_BALANCED,
    AGENT_SYSTEM_PROMPT_STRICT,
    INSUFFICIENT_INFO_MESSAGE,
    PARTIAL_ANSWER_PREFIX,
    format_node_for_prompt,
    format_nodes_for_prompt,
    get_agent_system_prompt,
    get_faithfulness_guard_prompt,
    get_refine_template,
    get_text_qa_template,
    resolve_grounding_mode,
)


def test_format_node_includes_metadata() -> None:
    node = NodeWithScore(
        node=TextNode(
            text="Employees receive three days of sick leave.",
            metadata={
                "source_file": "handbook.pdf",
                "page_number": 18,
                "section_path": "5.3 Sick Leave",
                "section_number": "5.3",
            },
        ),
        score=0.9,
    )
    formatted = format_node_for_prompt(node, 1)
    assert "<source id=\"1\">" in formatted
    assert "handbook.pdf" in formatted
    assert "p.18" in formatted
    assert "5.3 Sick Leave" in formatted
    assert "three days of sick leave" in formatted


def test_format_nodes_empty() -> None:
    assert "no document excerpts" in format_nodes_for_prompt([])


def test_strict_agent_prompt_forbids_inference() -> None:
    prompt = get_agent_system_prompt(strict=True)
    assert "Do NOT infer" in prompt or "Do NOT invent" in prompt


def test_balanced_agent_prompt_allows_synthesis() -> None:
    prompt = get_agent_system_prompt(strict=False, version="v2_balanced")
    assert prompt == AGENT_SYSTEM_PROMPT_BALANCED
    assert "synthesize" in prompt.lower() or "MAY synthesize" in prompt
    assert "[Source N]" in prompt
    assert "preserve them verbatim" in prompt


def test_strict_text_qa_template_has_few_shot() -> None:
    tmpl = get_text_qa_template(strict=True).template
    assert "Example A" in tmpl
    assert "Example B" in tmpl


def test_balanced_text_qa_allows_partial_answers() -> None:
    tmpl = get_text_qa_template(strict=False, version="v2_balanced").template
    assert PARTIAL_ANSWER_PREFIX in tmpl
    assert "synthesize" in tmpl.lower() or "SYNTHESIZE" in tmpl
    assert "Example B" in tmpl
    assert "Example E" in tmpl
    assert "Example J" in tmpl
    assert "health benefits" in tmpl.lower()
    assert "MUST answer with citations" in tmpl
    assert "CITATION RULES (MANDATORY)" in tmpl
    assert "MUST end with at least one" in tmpl
    assert "[Source N]" in tmpl
    assert "NEVER append the insufficient-information message" in tmpl
    assert "social media" in tmpl.lower()
    assert "Abstain ONLY" in tmpl


def test_balanced_refine_template_preserves_source_tags() -> None:
    tmpl = get_refine_template(strict=False, version="v2_balanced").template
    assert "Preserve all existing [Source N] tags" in tmpl
    assert "Add [Source N] tags" in tmpl


def test_balanced_guard_checks_unsupported_only() -> None:
    guard = get_faithfulness_guard_prompt(mode="balanced")
    assert "UNSUPPORTED" in guard
    assert "SUPPORTED" in guard


def test_resolve_grounding_mode() -> None:
    assert resolve_grounding_mode(strict=True) == "strict"
    assert resolve_grounding_mode(strict=False) == "balanced"
    assert resolve_grounding_mode(version="v2_strict") == "strict"
    assert resolve_grounding_mode(version="v2_balanced") == "balanced"


def test_standard_prompt_differs_from_strict() -> None:
    strict = get_agent_system_prompt(strict=True)
    standard = get_agent_system_prompt(strict=False, version="v1_standard")
    assert strict == AGENT_SYSTEM_PROMPT_STRICT
    assert strict != standard