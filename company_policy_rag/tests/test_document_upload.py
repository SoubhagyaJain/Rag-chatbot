"""Tests for legal PDF upload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_upload import (
    list_legal_documents,
    sanitize_upload_filename,
    save_legal_pdf,
    validate_pdf_content,
)

_MINIMAL_PDF = b"%PDF-1.4\n%EOF\n"


def test_sanitize_upload_filename_strips_path():
    assert sanitize_upload_filename("../../evil.pdf") == "evil.pdf"
    assert sanitize_upload_filename("My Contract (2024).pdf") == "My_Contract_2024.pdf"


def test_validate_pdf_content_rejects_empty_and_non_pdf():
    with pytest.raises(ValueError, match="empty"):
        validate_pdf_content(b"")
    with pytest.raises(ValueError, match="not a valid PDF"):
        validate_pdf_content(b"not-a-pdf")


def test_save_legal_pdf_writes_file(tmp_path, monkeypatch):
    import src.document_upload as du
    import src.config as cfg

    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    monkeypatch.setattr(cfg.settings, "legal_dir", legal_dir)
    monkeypatch.setattr(
        du,
        "settings",
        cfg.settings,
    )

    path = save_legal_pdf(_MINIMAL_PDF, "contract.pdf")
    assert path.exists()
    assert path.parent == legal_dir
    assert path.read_bytes() == _MINIMAL_PDF


def test_list_legal_documents_sorted(tmp_path, monkeypatch):
    import src.config as cfg

    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    (legal_dir / "b.pdf").write_bytes(_MINIMAL_PDF)
    (legal_dir / "a.pdf").write_bytes(_MINIMAL_PDF)
    monkeypatch.setattr(cfg.settings, "legal_dir", legal_dir)

    docs = list_legal_documents()
    assert [d["filename"] for d in docs] == ["a.pdf", "b.pdf"]
    assert docs[0]["size_kb"] >= 0