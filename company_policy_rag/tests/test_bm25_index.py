"""Tests for BM25 index build, search, and persistence."""

from __future__ import annotations

from llama_index.core.schema import TextNode

from src.bm25_index import (
    BM25Index,
    clear_bm25_storage,
    load_bm25_index,
    rebuild_bm25_index,
    reset_bm25_cache,
    save_bm25_index,
    tokenize,
)


def test_tokenize_lowercase_words() -> None:
    assert "flsa" in tokenize("FLSA overtime eligibility")


def test_bm25_search_exact_term() -> None:
    nodes = [
        TextNode(
            text="Non-exempt employees receive overtime under FLSA rules.",
            metadata={"source_file": "handbook.pdf", "section_title": "Overtime"},
            id_="n1",
        ),
        TextNode(
            text="Dress code requires business casual attire.",
            metadata={"source_file": "handbook.pdf"},
            id_="n2",
        ),
    ]
    index = BM25Index()
    index.build(nodes)
    hits = index.search("FLSA overtime", top_k=2)
    assert hits
    assert hits[0][0] == "n1"


def test_persist_and_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.bm25_index.settings.bm25_storage_dir", tmp_path)
    reset_bm25_cache()

    nodes = [
        TextNode(text="currency conversion tool example", metadata={}, id_="c1"),
    ]
    built = rebuild_bm25_index(nodes)
    assert built.size == 1

    reset_bm25_cache()
    loaded = load_bm25_index()
    assert loaded is not None
    assert loaded.size == 1
    hits = loaded.search("currency tool", top_k=1)
    assert hits[0][0] == "c1"


def test_remove_by_source_file() -> None:
    nodes = [
        TextNode(text="policy A", metadata={"source_file": "a.pdf"}, id_="a"),
        TextNode(text="policy B", metadata={"source_file": "b.pdf"}, id_="b"),
    ]
    index = BM25Index()
    index.build(nodes)
    index.remove_by_source_file("a.pdf")
    assert index.size == 1
    assert index.entries[0].node_id == "b"


def test_clear_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.bm25_index.settings.bm25_storage_dir", tmp_path)
    nodes = [TextNode(text="x", metadata={}, id_="x1")]
    rebuild_bm25_index(nodes)
    clear_bm25_storage()
    reset_bm25_cache()
    assert load_bm25_index() is None