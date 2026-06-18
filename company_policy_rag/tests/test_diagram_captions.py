"""Tests for diagram caption node generation."""

from __future__ import annotations

import json
from pathlib import Path

from llama_index.core.schema import Document

from src.diagram_captions import _heuristic_caption, build_caption_nodes


class TestDiagramCaptions:
    def test_heuristic_caption_from_figure_line(self) -> None:
        context = "Some intro\nFigure 3: Agent workflow with retrieval step\nMore text"
        caption = _heuristic_caption(context, source_file="guide.pdf", page_number=12)
        assert "DIAGRAM CAPTION" in caption
        assert "Figure 3" in caption

    def test_build_nodes_from_manifest(self, tmp_path, monkeypatch) -> None:
        images_dir = tmp_path / "images"
        file_hash = "abc123"
        page_dir = images_dir / file_hash
        page_dir.mkdir(parents=True)
        manifest = {
            "pages": {
                "2": {"embedded": ["img1.png"], "thumbnail": "thumb.png"},
            }
        }
        (page_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        monkeypatch.setattr("src.diagram_captions.settings.pdf_images_dir", images_dir)
        monkeypatch.setattr("src.diagram_captions.settings.enable_diagram_captions", True)
        monkeypatch.setattr("src.diagram_captions.settings.enable_caption_llm", False)

        docs = [
            Document(
                text="Figure 2: Multi-agent orchestration diagram",
                metadata={
                    "source_file": "AI_Agents_guidebook.pdf",
                    "page_number": 2,
                    "file_hash": file_hash,
                    "document_type": "legal_document",
                },
            )
        ]
        nodes = build_caption_nodes(
            docs, file_hash=file_hash, source_file="AI_Agents_guidebook.pdf"
        )
        assert len(nodes) == 1
        assert nodes[0].metadata.get("content_type") == "diagram_caption"

    def test_caption_inherits_section_metadata(self, tmp_path, monkeypatch) -> None:
        images_dir = tmp_path / "images"
        file_hash = "def456"
        page_dir = images_dir / file_hash
        page_dir.mkdir(parents=True)
        manifest = {"pages": {"11": {"thumbnail": "thumb.png"}}}
        (page_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        monkeypatch.setattr("src.diagram_captions.settings.pdf_images_dir", images_dir)
        monkeypatch.setattr("src.diagram_captions.settings.enable_diagram_captions", True)
        monkeypatch.setattr("src.diagram_captions.settings.enable_caption_llm", False)

        docs = [
            Document(
                text="6. Memory\n\nAgents need memory for context.",
                metadata={
                    "source_file": "AI_Agents_guidebook.pdf",
                    "page_number": 11,
                    "file_hash": file_hash,
                    "section_path": "6. Memory",
                    "section_title": "Memory",
                    "section_number": "6",
                },
            )
        ]
        nodes = build_caption_nodes(
            docs, file_hash=file_hash, source_file="AI_Agents_guidebook.pdf"
        )
        assert len(nodes) == 1
        assert nodes[0].metadata.get("section_path") == "6. Memory"