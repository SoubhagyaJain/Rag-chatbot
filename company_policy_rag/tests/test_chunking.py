"""Tests for hierarchical chunking and code-block protection."""

from __future__ import annotations

from llama_index.core.schema import Document

from src.chunking import _split_into_segments, documents_to_hierarchical_nodes


class TestCodeBlockProtection:
    def test_split_preserves_fenced_code(self) -> None:
        text = "Before\n```python\nx = 1\ny = 2\n```\nAfter"
        segments = _split_into_segments(text)
        code_segments = [s for t, s in segments if t == "code"]
        assert len(code_segments) == 1
        assert "x = 1" in code_segments[0]
        assert "y = 2" in code_segments[0]

    def test_oversized_code_is_atomic_child(self) -> None:
        code_body = "```python\n" + "print('line')\n" * 80 + "```"
        doc = Document(
            text=f"Intro\n{code_body}\nOutro",
            metadata={
                "source_file": "guide.pdf",
                "page_number": 5,
                "section_path": "Tools",
            },
        )
        result = documents_to_hierarchical_nodes([doc], persist_parents=False)
        code_children = [
            n for n in result.embed_nodes if (n.metadata or {}).get("content_type") == "code"
        ]
        assert len(code_children) >= 1
        assert all((n.metadata or {}).get("is_atomic") for n in code_children)
        assert "[CODE BLOCK" in (code_children[0].text or "")

    def test_parent_child_linkage(self) -> None:
        doc = Document(
            text="Section about vacation accrual for full-time staff.",
            metadata={
                "source_file": "handbook.pdf",
                "page_number": 1,
                "section_path": "Leave",
            },
        )
        result = documents_to_hierarchical_nodes([doc], persist_parents=False)
        assert len(result.parent_nodes) == 1
        parent_id = result.parent_nodes[0].node_id
        assert all((n.metadata or {}).get("parent_id") == parent_id for n in result.embed_nodes)