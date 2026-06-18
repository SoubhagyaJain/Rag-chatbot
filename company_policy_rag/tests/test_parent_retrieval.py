"""Tests for parent-document expansion."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.docstore import register_parent_nodes
from src.parent_retrieval import expand_to_parent_documents


def test_expand_child_to_parent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.docstore.settings.docstore_dir", tmp_path)
    monkeypatch.setattr("src.parent_retrieval.settings.enable_parent_document_retrieval", True)

    parent = TextNode(
        text="Full parent section with vacation, sick leave, and benefits details " * 20,
        metadata={"source_file": "handbook.pdf", "child_ids": ["child_1"]},
        id_="parent_abc",
    )
    register_parent_nodes([parent])

    child = NodeWithScore(
        node=TextNode(
            text="Short child about vacation only.",
            metadata={
                "parent_id": "parent_abc",
                "node_role": "child",
                "page_number": 14,
                "section_path": "Leave > Vacation",
                "source_file": "handbook.pdf",
            },
            id_="child_1",
        ),
        score=0.85,
    )

    expanded = expand_to_parent_documents([child])
    assert len(expanded) == 1
    assert len(expanded[0].get_content() or "") > len(child.get_content() or "")
    assert expanded[0].metadata.get("expanded_from_child") is True
    assert expanded[0].metadata.get("page_number") == 14


def test_caption_passthrough(monkeypatch) -> None:
    monkeypatch.setattr("src.parent_retrieval.settings.enable_parent_document_retrieval", True)
    caption = NodeWithScore(
        node=TextNode(
            text="[DIAGRAM CAPTION] Agent workflow",
            metadata={"node_role": "caption", "content_type": "diagram_caption"},
            id_="cap_1",
        ),
        score=0.7,
    )
    expanded = expand_to_parent_documents([caption])
    assert expanded[0].node.node_id == "cap_1"


def test_dedup_same_parent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.docstore.settings.docstore_dir", tmp_path)
    monkeypatch.setattr("src.parent_retrieval.settings.enable_parent_document_retrieval", True)

    parent = TextNode(
        text="Parent text " * 50,
        metadata={"source_file": "g.pdf"},
        id_="parent_x",
    )
    register_parent_nodes([parent])

    children = [
        NodeWithScore(
            node=TextNode(
                text="child a",
                metadata={"parent_id": "parent_x", "node_role": "child"},
                id_="c_a",
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                text="child b",
                metadata={"parent_id": "parent_x", "node_role": "child"},
                id_="c_b",
            ),
            score=0.8,
        ),
    ]
    expanded = expand_to_parent_documents(children)
    assert len(expanded) == 1
    assert expanded[0].node.node_id == "parent_x"