"""Tests for PDF parser routing and block detection."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import Document

from src.pdf_parsers import _detect_block_types, resolve_pdf_parser


class TestBlockDetection:
    def test_fenced_code_detected(self) -> None:
        text = "Intro\n\n```python\ndef foo():\n    return 1\n```\n\nAfter"
        blocks = _detect_block_types(text)
        types = [b["block_type"] for b in blocks]
        assert "code" in types
        assert "text" in types

    def test_prose_only(self) -> None:
        blocks = _detect_block_types("Plain policy paragraph about vacation accrual.")
        assert blocks[0]["block_type"] == "text"


class TestParserResolution:
    def test_default_is_pypdf_when_marker_disabled(self) -> None:
        assert resolve_pdf_parser() == "pypdf"


class TestPypdfLoad:
    def test_load_handbook_pdf(self) -> None:
        pdf = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "policies"
            / "Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf"
        )
        if not pdf.exists():
            return
        from src.pdf_parsers import load_pdf_as_documents

        docs = load_pdf_as_documents(pdf, base_metadata={"source_file": pdf.name})
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].metadata.get("parser") == "pypdf"