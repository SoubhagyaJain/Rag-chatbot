"""Unit tests for section detection, metadata, and citation utilities."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import TextNode

from src.utils import (
    SectionTracker,
    dedupe_citations,
    detect_section_title,
    enrich_text_with_section_context,
    format_citation,
    format_citation_primary,
    infer_category,
    parse_section_heading,
    prepare_citations_for_display,
    scan_text_for_headings,
    shorten_source_filename,
)


class TestSectionDetection:
    def test_numbered_section(self) -> None:
        text = "5.2 Vacation Benefits\n\nEmployees accrue vacation..."
        heading = parse_section_heading("5.2 Vacation Benefits")
        assert heading is not None
        assert heading.section_number == "5.2"
        assert heading.section_title == "Vacation Benefits"
        assert detect_section_title(text) == "5.2 Vacation Benefits"

    def test_roman_numeral(self) -> None:
        heading = parse_section_heading("II. GENERAL EMPLOYMENT INFORMATION")
        assert heading is not None
        assert heading.section_number == "II"
        assert heading.level == 1

    def test_letter_subsection(self) -> None:
        heading = parse_section_heading("A. Employment At-Will")
        assert heading is not None
        assert heading.section_number == "A"
        assert heading.section_title == "Employment At-Will"
        assert heading.level == 2

    def test_article_heading(self) -> None:
        heading = parse_section_heading("Article 7: Confidentiality Obligations")
        assert heading is not None
        assert heading.section_number == "7"
        assert "Confidentiality" in heading.section_title

    def test_section_decimal(self) -> None:
        heading = parse_section_heading("Section 3.1 — Harassment Policy")
        assert heading is not None
        assert heading.section_number == "3.1"

    def test_all_caps_heading_standard_mode(self) -> None:
        heading = parse_section_heading("EQUAL EMPLOYMENT OPPORTUNITY")
        assert heading is not None
        assert heading.section_title == "EQUAL EMPLOYMENT OPPORTUNITY"

    def test_no_heading(self) -> None:
        text = "This paragraph has no clear section header."
        assert detect_section_title(text) is None

    def test_noise_filtered(self) -> None:
        assert parse_section_heading("Page 12") is None
        assert parse_section_heading("Table of Contents") is None


class TestSectionTracker:
    def test_hierarchical_path(self) -> None:
        tracker = SectionTracker()
        tracker.update(parse_section_heading("II. GENERAL EMPLOYMENT INFORMATION"))  # type: ignore[arg-type]
        tracker.update(parse_section_heading("A. Employment At-Will"))  # type: ignore[arg-type]

        ctx = tracker.current_context()
        assert ctx.section_title == "Employment At-Will"
        assert ctx.section_number == "A"
        assert ctx.section_path == (
            "II. GENERAL EMPLOYMENT INFORMATION > A. Employment At-Will"
        )

    def test_roman_resets_deeper_levels(self) -> None:
        tracker = SectionTracker()
        tracker.update(parse_section_heading("I. INTRODUCTION"))  # type: ignore[arg-type]
        tracker.update(parse_section_heading("A. Purpose"))  # type: ignore[arg-type]
        tracker.update(parse_section_heading("II. EMPLOYMENT"))  # type: ignore[arg-type]

        ctx = tracker.current_context()
        assert "A." not in (ctx.section_path or "")
        assert ctx.section_number == "II"

    def test_propagation_through_body_text(self) -> None:
        tracker = SectionTracker()
        enrich_text_with_section_context(
            "II. BENEFITS\n\nSome intro text.",
            tracker,
        )
        # Continuation page with no heading — tracker should retain context
        ctx = enrich_text_with_section_context(
            "Vacation accrues monthly for all full-time employees.",
            tracker,
        )
        assert ctx.section_title == "BENEFITS"
        assert ctx.section_number == "II"

    def test_mid_text_heading_updates_path(self) -> None:
        tracker = SectionTracker()
        enrich_text_with_section_context("II. GENERAL EMPLOYMENT INFORMATION\n", tracker)
        enrich_text_with_section_context(
            "B. Equal Employment Opportunity\n\nWe are an equal opportunity employer.",
            tracker,
        )
        ctx = tracker.current_context()
        assert "B. Equal Employment Opportunity" in (ctx.section_path or "")


class TestHandbookScenarios:
    def test_vacation_accrual_section(self) -> None:
        text = """
II. GENERAL EMPLOYMENT INFORMATION

A. Employment At-Will

B. Equal Employment Opportunity

5.2 Vacation Benefits

Full-time employees accrue vacation at a rate of 1.25 days per month.
"""
        headings = scan_text_for_headings(text)
        labels = [h.full_label for h in headings]
        assert "II. GENERAL EMPLOYMENT INFORMATION" in labels
        assert "A. Employment At-Will" in labels
        assert "5.2 Vacation Benefits" in labels

    def test_harassment_section(self) -> None:
        text = """
Section 3.1 — Harassment Policy

The company prohibits all forms of harassment in the workplace.
"""
        heading = parse_section_heading("Section 3.1 — Harassment Policy")
        assert heading is not None
        assert "Harassment" in heading.section_title


class TestCategoryInference:
    def test_policy_category(self) -> None:
        path = Path("employee_handbook.pdf")
        cat = infer_category(path, "company_policy")
        assert cat == "Policy — Employee Handbook"

    def test_legal_category(self) -> None:
        path = Path("nda_template.pdf")
        cat = infer_category(path, "legal_document")
        assert cat == "Legal — Nda Template"


class TestCitationDisplay:
    def test_shorten_source_filename(self) -> None:
        assert shorten_source_filename("employee_handbook.pdf") == "Employee Handbook"

    def test_format_citation_primary_section_first(self) -> None:
        citation = {
            "source_file": "employee_handbook.pdf",
            "page_number": 14,
            "section_path": "II. GENERAL EMPLOYMENT INFORMATION > 5.2 Vacation Benefits",
            "section_number": "5.2",
        }
        label = format_citation_primary(citation, fmt="section_first")
        assert "5.2 Vacation Benefits" in label
        assert "(p.14)" in label

    def test_format_citation_primary_document_first(self) -> None:
        citation = {
            "source_file": "employee_handbook.pdf",
            "page_number": 14,
            "section_title": "Vacation Benefits",
            "section_number": "5.2",
        }
        label = format_citation_primary(citation, fmt="document_first")
        assert "Employee Handbook" in label
        assert "Section 5.2" in label
        assert "(Page 14)" in label

    def test_format_citation_fallback_no_section(self) -> None:
        citation = {
            "source_file": "misc_policy.pdf",
            "page_number": 3,
        }
        label = format_citation_primary(citation, fmt="section_first")
        assert "Misc Policy" in label
        assert "(p.3)" in label

    def test_dedupe_citations(self) -> None:
        citations = [
            {"source_file": "a.pdf", "page_number": 1, "section_path": "A", "score": 0.5},
            {"source_file": "a.pdf", "page_number": 1, "section_path": "A", "score": 0.9},
            {"source_file": "a.pdf", "page_number": 2, "section_path": "B", "score": 0.7},
        ]
        deduped = dedupe_citations(citations)
        assert len(deduped) == 2
        kept = next(c for c in deduped if c["section_path"] == "A")
        assert kept["score"] == 0.9

    def test_prepare_citations_caps_count(self) -> None:
        citations = [
            {"source_file": f"f{i}.pdf", "page_number": i, "score": 0.1 * i}
            for i in range(1, 12)
        ]
        prepared = prepare_citations_for_display(citations)
        assert len(prepared) <= 6


class TestCitationFormatting:
    def test_format_citation_with_section_path(self) -> None:
        node = TextNode(
            text="Employees receive 15 days of PTO annually.",
            metadata={
                "source_file": "employee_handbook.pdf",
                "page_number": 12,
                "section_title": "Vacation Benefits",
                "section_number": "5.2",
                "section_path": (
                    "II. GENERAL EMPLOYMENT INFORMATION > 5.2 Vacation Benefits"
                ),
                "document_type": "company_policy",
                "category": "Policy — Employee Handbook",
            },
        )
        citation = format_citation(node)
        assert citation["source_file"] == "employee_handbook.pdf"
        assert citation["page_number"] == 12
        assert citation["section_number"] == "5.2"
        assert "5.2 Vacation Benefits" in citation["section_path"]
        assert "PTO" in citation["excerpt"]