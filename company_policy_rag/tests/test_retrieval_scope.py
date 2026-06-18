"""Tests for corpus-scoped retrieval filters."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval_scope import (
    ScopeFilteredRetriever,
    corpus_retrieval_filters,
    filter_nodes_by_metadata,
    node_matches_filters,
)


def _node(source_file: str, document_type: str = "") -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text="chunk",
            metadata={"source_file": source_file, "document_type": document_type},
        ),
        score=0.5,
    )


class TestCorpusRetrievalFilters:
    def test_guidebook_filter_by_source(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.retrieval_scope.settings.enable_corpus_scoped_retrieval", True
        )
        f = corpus_retrieval_filters("guidebook")
        assert f == {"source_file": "AI_Agents_guidebook.pdf"}

    def test_policy_filter_by_source(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.retrieval_scope.settings.enable_corpus_scoped_retrieval", True
        )
        f = corpus_retrieval_filters("policy")
        assert "Employee-Handbook" in f["source_file"]

    def test_disabled_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.retrieval_scope.settings.enable_corpus_scoped_retrieval", False
        )
        assert corpus_retrieval_filters("guidebook") is None

    def test_explicit_source_file_override(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.retrieval_scope.settings.enable_corpus_scoped_retrieval", True
        )
        f = corpus_retrieval_filters(
            "guidebook", source_file="custom.pdf"
        )
        assert f == {"source_file": "custom.pdf"}


class TestNodeFiltering:
    def test_filter_removes_wrong_source(self) -> None:
        nodes = [
            _node("AI_Agents_guidebook.pdf"),
            _node("Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf"),
        ]
        filtered = filter_nodes_by_metadata(
            nodes, {"source_file": "AI_Agents_guidebook.pdf"}
        )
        assert len(filtered) == 1
        assert filtered[0].metadata["source_file"] == "AI_Agents_guidebook.pdf"

    def test_scope_filtered_retriever_wrapper(self) -> None:
        class FakeInner:
            def retrieve(self, _query: str) -> list[NodeWithScore]:
                return [
                    _node("AI_Agents_guidebook.pdf"),
                    _node("Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf"),
                ]

        wrapped = ScopeFilteredRetriever(
            FakeInner(), {"source_file": "AI_Agents_guidebook.pdf"}
        )
        result = wrapped.retrieve("test")
        assert len(result) == 1
        assert node_matches_filters(result[0], {"source_file": "AI_Agents_guidebook.pdf"})