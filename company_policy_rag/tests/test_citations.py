"""Tests for citation selection (generation-linked sources only)."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.citations import (
    extract_cited_source_indices,
    filter_nodes_by_relevance_score,
    select_citations_for_answer,
)


def _node(section: str, score: float, page: int = 1) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=f"Policy text for {section}",
            metadata={
                "section_path": section,
                "section_title": section.split(">")[-1].strip(),
                "page_number": page,
                "source_file": "handbook.pdf",
            },
        ),
        score=score,
    )


def test_extract_cited_source_indices_single_and_multi() -> None:
    answer = "Employees receive sick leave [Source 1]. See also [Source 2, Source 3]."
    assert extract_cited_source_indices(answer) == {1, 2, 3}


def test_select_citations_only_cited_sources() -> None:
    nodes = [
        _node("IV. LEAVE > Sick Leave", 8.0, 20),
        _node("IV. LEAVE > Holidays", 5.0, 22),
        _node("V. CONDUCT > Visitors", 4.0, 30),
    ]
    answer = (
        "Based on the available information, employees receive three days of "
        "paid sick leave [Source 1]."
    )
    citations = select_citations_for_answer(answer, nodes)
    assert len(citations) == 1
    assert "Sick Leave" in citations[0]["section_path"]
    assert citations[0]["selection_reason"] == "cited_in_answer"


def test_select_citations_excludes_uncited_generation_nodes() -> None:
    nodes = [
        _node("IV. LEAVE > Sick Leave", 8.0),
        _node("IV. LEAVE > Family Care Leave", 7.5),
        _node("IV. LEAVE > Holidays", 6.0),
    ]
    answer = "Sick leave policy details [Source 1]."
    citations = select_citations_for_answer(answer, nodes)
    paths = [c["section_path"] for c in citations]
    assert any("Sick Leave" in p for p in paths)
    assert not any("Holidays" in p for p in paths)
    assert not any("Family Care" in p for p in paths)


def test_filter_nodes_by_relevance_score() -> None:
    nodes = [
        _node("A", 10.0),
        _node("B", 6.0),
        _node("C", 3.0),
    ]
    filtered = filter_nodes_by_relevance_score(nodes, min_ratio=0.55)
    paths = [n.metadata["section_path"] for n in filtered]
    assert "A" in paths
    assert "B" in paths
    assert "C" not in paths


def test_select_citations_score_fallback_when_no_tags() -> None:
    nodes = [
        _node("IV. LEAVE > Sick Leave", 9.0),
        _node("IV. LEAVE > Holidays", 4.0),
    ]
    answer = "Employees receive paid sick leave after 120 days."
    citations = select_citations_for_answer(answer, nodes)
    assert len(citations) >= 1
    assert citations[0]["selection_reason"] == "score_threshold_fallback"
    assert "Sick Leave" in citations[0]["section_path"]