"""Tests for parent docstore persistence."""

from __future__ import annotations

from llama_index.core.schema import TextNode

from src.docstore import (
    get_parent_node,
    load_parent_store,
    register_parent_nodes,
    remove_parents_for_source,
)


class TestDocstore:
    def test_register_and_fetch_parent(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.docstore.settings.docstore_dir", tmp_path)
        parent = TextNode(
            text="Parent section text",
            metadata={"source_file": "test.pdf", "child_ids": ["c1"], "node_role": "parent"},
            id_="parent_test123",
        )
        register_parent_nodes([parent])
        store = load_parent_store()
        assert "parent_test123" in store["parents"]
        fetched = get_parent_node("parent_test123")
        assert fetched is not None
        assert "Parent section" in (fetched.text or "")

    def test_remove_by_source(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.docstore.settings.docstore_dir", tmp_path)
        parent = TextNode(
            text="Remove me",
            metadata={"source_file": "gone.pdf", "child_ids": []},
            id_="parent_rm",
        )
        register_parent_nodes([parent])
        removed = remove_parents_for_source("gone.pdf")
        assert removed == 1
        assert get_parent_node("parent_rm") is None