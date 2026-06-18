"""Tests for PDF image extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pdf_images import (
    MANIFEST_FILE,
    extract_pdf_images,
    get_page_images,
    images_dir_for_hash,
    remove_images_for_source,
)


def _write_manifest(images_dir: Path, *, source_file: str, file_hash: str, pages: dict) -> None:
    dest = images_dir / file_hash
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "page_2_img_0.png").write_bytes(b"png")
    (dest / "page_2_thumb.png").write_bytes(b"thumb")
    manifest = {
        "source_file": source_file,
        "file_hash": file_hash,
        "pages": pages,
        "embedded_count": 1,
    }
    (dest / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    registry = images_dir / "by_source.json"
    registry.write_text(json.dumps({source_file: file_hash}), encoding="utf-8")


def test_get_page_images_returns_embedded_and_thumbnail(tmp_path, monkeypatch):
    import src.config as cfg

    images_dir = tmp_path / "images"
    monkeypatch.setattr(cfg.settings, "pdf_images_dir", images_dir)
    monkeypatch.setattr(cfg.settings, "enable_pdf_images", True)

    _write_manifest(
        images_dir,
        source_file="contract.pdf",
        file_hash="abc123",
        pages={"2": {"embedded": ["page_2_img_0.png"], "thumbnail": "page_2_thumb.png"}},
    )

    paths = get_page_images("contract.pdf", 2)
    assert [p.name for p in paths] == ["page_2_img_0.png", "page_2_thumb.png"]


def test_get_page_images_empty_when_disabled(tmp_path, monkeypatch):
    import src.config as cfg

    images_dir = tmp_path / "images"
    monkeypatch.setattr(cfg.settings, "pdf_images_dir", images_dir)
    monkeypatch.setattr(cfg.settings, "enable_pdf_images", False)

    _write_manifest(
        images_dir,
        source_file="contract.pdf",
        file_hash="abc123",
        pages={"2": {"embedded": ["page_2_img_0.png"], "thumbnail": "page_2_thumb.png"}},
    )

    assert get_page_images("contract.pdf", 2) == []


def test_remove_images_for_source_deletes_cache(tmp_path, monkeypatch):
    import src.config as cfg

    images_dir = tmp_path / "images"
    monkeypatch.setattr(cfg.settings, "pdf_images_dir", images_dir)

    _write_manifest(
        images_dir,
        source_file="contract.pdf",
        file_hash="abc123",
        pages={"1": {"embedded": ["page_2_img_0.png"], "thumbnail": None}},
    )

    remove_images_for_source("contract.pdf")

    assert not images_dir_for_hash("abc123").exists()
    assert json.loads((images_dir / "by_source.json").read_text(encoding="utf-8")) == {}


def test_extract_pdf_images_skips_when_manifest_unchanged(tmp_path, monkeypatch):
    import src.config as cfg

    images_dir = tmp_path / "images"
    monkeypatch.setattr(cfg.settings, "pdf_images_dir", images_dir)
    monkeypatch.setattr(cfg.settings, "enable_pdf_images", True)

    _write_manifest(
        images_dir,
        source_file="sample.pdf",
        file_hash="hash1",
        pages={"1": {"embedded": ["page_2_img_0.png"], "thumbnail": "page_2_thumb.png"}},
    )

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with patch("fitz.open") as mock_open:
        manifest = extract_pdf_images(
            pdf_path,
            file_hash="hash1",
            source_file="sample.pdf",
            force=False,
        )

    mock_open.assert_not_called()
    assert manifest["file_hash"] == "hash1"
    assert manifest["embedded_count"] == 1


def test_extract_pdf_images_disabled_returns_empty(tmp_path, monkeypatch):
    import src.config as cfg

    monkeypatch.setattr(cfg.settings, "enable_pdf_images", False)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    assert extract_pdf_images(pdf_path, file_hash="hash1") == {}