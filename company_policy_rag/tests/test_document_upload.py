"""Tests for legal PDF upload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_upload import (
    list_legal_documents,
    remove_legal_document,
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


def test_remove_legal_document_deletes_file(tmp_path, monkeypatch):
    import src.config as cfg
    import src.document_upload as du

    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    pdf_path = legal_dir / "contract.pdf"
    pdf_path.write_bytes(_MINIMAL_PDF)
    monkeypatch.setattr(cfg.settings, "legal_dir", legal_dir)
    monkeypatch.setattr(du, "settings", cfg.settings)
    monkeypatch.setattr(du, "configure_llama_index", lambda: None)
    monkeypatch.setattr(du, "remove_document_from_index", lambda _name: 3)
    monkeypatch.setattr(du, "remove_images_for_source", lambda _name: None)

    result = remove_legal_document("contract.pdf")

    assert result.filename == "contract.pdf"
    assert result.file_deleted is True
    assert result.chunks_removed == 3
    assert not pdf_path.exists()


def test_remove_legal_document_rejects_path_traversal(tmp_path, monkeypatch):
    import src.config as cfg
    import src.document_upload as du

    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    monkeypatch.setattr(cfg.settings, "legal_dir", legal_dir)
    monkeypatch.setattr(du, "settings", cfg.settings)

    with pytest.raises(ValueError, match="Invalid filename"):
        remove_legal_document("../secret.pdf")


def test_remove_legal_document_missing_file(tmp_path, monkeypatch):
    import src.config as cfg
    import src.document_upload as du

    legal_dir = tmp_path / "legal"
    legal_dir.mkdir()
    monkeypatch.setattr(cfg.settings, "legal_dir", legal_dir)
    monkeypatch.setattr(du, "settings", cfg.settings)

    with pytest.raises(FileNotFoundError, match="not found"):
        remove_legal_document("missing.pdf")