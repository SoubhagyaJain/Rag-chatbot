"""
In-memory BM25 index for hybrid lexical + dense retrieval.

Persisted under storage/bm25/ and rebuilt on indexing. At ~300 chunks this is
fast enough without Elasticsearch; exact legal terms (FLSA, clause IDs) benefit.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode

from src.config import settings
from src.utils import logger

CORPUS_FILE = "corpus.json"
INDEX_FILE = "index.pkl"

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def _searchable_text(node: TextNode) -> str:
    meta = node.metadata or {}
    parts = [
        node.text or "",
        str(meta.get("section_path", "")),
        str(meta.get("section_title", "")),
        str(meta.get("section_number", "")),
        str(meta.get("source_file", "")),
        str(meta.get("content_type", "")),
    ]
    return " ".join(p for p in parts if p).strip()


@dataclass
class BM25CorpusEntry:
    node_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25Index:
    """BM25 index over embeddable chunks."""

    entries: list[BM25CorpusEntry] = field(default_factory=list)
    _bm25: Any = None
    _tokenized_corpus: list[list[str]] = field(default_factory=list)

    def build(self, nodes: list[TextNode]) -> None:
        from rank_bm25 import BM25Okapi

        self.entries = []
        self._tokenized_corpus = []
        for node in nodes:
            node_id = node.node_id
            if not node_id:
                continue
            text = _searchable_text(node)
            if not text:
                continue
            self.entries.append(
                BM25CorpusEntry(
                    node_id=node_id,
                    text=text,
                    metadata=dict(node.metadata or {}),
                )
            )
            self._tokenized_corpus.append(tokenize(text))

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._bm25 or not self.entries:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        # Return top ranks even when raw scores are 0 (common on small corpora).
        # Hybrid RRF uses rank position, not absolute BM25 score.
        return [(self.entries[idx].node_id, float(score)) for idx, score in ranked]

    def get_entry(self, node_id: str) -> BM25CorpusEntry | None:
        for entry in self.entries:
            if entry.node_id == node_id:
                return entry
        return None

    def remove_by_source_file(self, source_file: str) -> None:
        kept_entries: list[BM25CorpusEntry] = []
        kept_tokens: list[list[str]] = []
        for entry, tokens in zip(self.entries, self._tokenized_corpus):
            if entry.metadata.get("source_file") == source_file:
                continue
            kept_entries.append(entry)
            kept_tokens.append(tokens)
        self.entries = kept_entries
        self._tokenized_corpus = kept_tokens
        if kept_tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(kept_tokens)
        else:
            self._bm25 = None

    @property
    def size(self) -> int:
        return len(self.entries)


def _storage_dir() -> Path:
    return settings.bm25_storage_dir


def _corpus_path() -> Path:
    return _storage_dir() / CORPUS_FILE


def _index_path() -> Path:
    return _storage_dir() / INDEX_FILE


def clear_bm25_storage() -> None:
    import shutil

    path = _storage_dir()
    if path.exists():
        shutil.rmtree(path)
    logger.info("Cleared BM25 storage at %s", path)


def save_bm25_index(index: BM25Index) -> None:
    _storage_dir().mkdir(parents=True, exist_ok=True)
    corpus_data = [
        {"node_id": e.node_id, "text": e.text, "metadata": e.metadata}
        for e in index.entries
    ]
    _corpus_path().write_text(json.dumps(corpus_data, indent=2), encoding="utf-8")
    with open(_index_path(), "wb") as f:
        pickle.dump(
            {"tokenized_corpus": index._tokenized_corpus, "entries": index.entries},
            f,
        )
    logger.info("Saved BM25 index (%d docs) to %s", index.size, _storage_dir())


def load_bm25_index() -> BM25Index | None:
    corpus_path = _corpus_path()
    index_path = _index_path()
    if not corpus_path.is_file():
        return None

    try:
        corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt BM25 corpus at %s", corpus_path)
        return None

    index = BM25Index()
    index.entries = [
        BM25CorpusEntry(
            node_id=item["node_id"],
            text=item["text"],
            metadata=item.get("metadata", {}),
        )
        for item in corpus_data
    ]

    if index_path.is_file():
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            index._tokenized_corpus = data.get("tokenized_corpus", [])
            if index._tokenized_corpus:
                from rank_bm25 import BM25Okapi

                index._bm25 = BM25Okapi(index._tokenized_corpus)
            return index
        except Exception as exc:
            logger.warning("BM25 pickle load failed (%s) — rebuilding from corpus", exc)

    # Rebuild BM25 from stored text
    index._tokenized_corpus = [tokenize(e.text) for e in index.entries]
    if index._tokenized_corpus:
        from rank_bm25 import BM25Okapi

        index._bm25 = BM25Okapi(index._tokenized_corpus)
    return index


# Lazy singleton for retrieval
_cached_index: BM25Index | None = None
_cache_loaded: bool = False


def reset_bm25_cache() -> None:
    global _cached_index, _cache_loaded
    _cached_index = None
    _cache_loaded = False


def get_bm25_index() -> BM25Index | None:
    """Return loaded BM25 index, or None if missing/disabled."""
    global _cached_index, _cache_loaded
    if not settings.enable_hybrid_bm25:
        return None
    if _cache_loaded:
        return _cached_index
    _cached_index = load_bm25_index()
    _cache_loaded = True
    return _cached_index


def rebuild_bm25_index(nodes: list[TextNode], *, merge: bool = False) -> BM25Index:
    """
    Build and persist BM25 from TextNodes.

    merge=True: append/replace into existing index (incremental indexing).
    """
    index = BM25Index()
    if merge:
        existing = load_bm25_index()
        if existing and existing.entries:
            # Full rebuild from existing + new is safer than partial merge
            all_entries = {e.node_id: e for e in existing.entries}
            for node in nodes:
                if node.node_id:
                    text = _searchable_text(node)
                    if text:
                        all_entries[node.node_id] = BM25CorpusEntry(
                            node_id=node.node_id,
                            text=text,
                            metadata=dict(node.metadata or {}),
                        )
            dummy_nodes = [
                TextNode(text=e.text, metadata=e.metadata, id_=e.node_id)
                for e in all_entries.values()
            ]
            index.build(dummy_nodes)
        else:
            index.build(nodes)
    else:
        index.build(nodes)

    save_bm25_index(index)
    reset_bm25_cache()
    global _cached_index, _cache_loaded
    _cached_index = index
    _cache_loaded = True
    return index


def rebuild_bm25_from_chroma() -> BM25Index | None:
    """Rebuild BM25 corpus from all Chroma chunks (recovery path)."""
    from src.indexing import get_chroma_collection

    if not index_exists_check():
        return None

    collection = get_chroma_collection()
    if collection.count() == 0:
        return None

    result = collection.get(include=["documents", "metadatas"])
    nodes: list[TextNode] = []
    for doc_id, text, meta in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        nodes.append(TextNode(text=text or "", metadata=meta or {}, id_=doc_id))

    return rebuild_bm25_index(nodes, merge=False)


def index_exists_check() -> bool:
    from src.indexing import index_exists

    return index_exists()


def remove_bm25_for_source(source_file: str) -> None:
    index = load_bm25_index()
    if not index:
        return
    index.remove_by_source_file(source_file)
    save_bm25_index(index)
    reset_bm25_cache()


def sync_bm25_with_chroma() -> BM25Index | None:
    """Rebuild BM25 if corpus size mismatches Chroma chunk count."""
    from src.indexing import get_collection_stats

    stats = get_collection_stats()
    chroma_count = stats.get("count", 0)
    if chroma_count <= 0:
        return None

    index = get_bm25_index()
    if index and index.size == chroma_count:
        return index

    logger.info(
        "BM25 corpus size (%s) != Chroma count (%d) — rebuilding",
        index.size if index else 0,
        chroma_count,
    )
    return rebuild_bm25_from_chroma()


def get_bm25_corpus_size() -> int:
    index = get_bm25_index()
    return index.size if index else 0