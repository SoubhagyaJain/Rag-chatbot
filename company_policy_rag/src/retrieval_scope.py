"""
Corpus-scoped retrieval filters to prevent cross-corpus chunk bleed.

Maps eval corpus labels (policy / guidebook) to Chroma metadata filters and
post-filters hybrid BM25 hits that fall outside the scope.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore

from src.config import settings

_CORPUS_DOCUMENT_TYPE: dict[str, str] = {
    "policy": "company_policy",
    "guidebook": "legal_document",
}

_CORPUS_DEFAULT_SOURCE_FILE: dict[str, str] = {
    "policy": "Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf",
    "guidebook": "AI_Agents_guidebook.pdf",
}


def resolve_query_filters(query: str, sidebar_scope: str | None) -> dict[str, str] | None:
    """
    Choose Chroma metadata filters for a query.

    When sidebar scope is 'all', auto-route handbook vs guidebook questions
    to prevent cross-corpus noise (e.g. dress code buried under guidebook hits).
    """
    scope = (sidebar_scope or "all").lower().strip()
    if scope in ("policy", "guidebook"):
        return corpus_retrieval_filters(scope)
    if scope != "all":
        return corpus_retrieval_filters(scope)

    from src.query_processing import detect_query_corpus

    inferred = detect_query_corpus(query)
    if inferred:
        return corpus_retrieval_filters(inferred)
    return None


def corpus_retrieval_filters(
    corpus: str | None,
    *,
    source_file: str | None = None,
) -> dict[str, str] | None:
    """
    Build Chroma metadata filters for a golden-case corpus.

    Prefers explicit source_file; falls back to corpus defaults.
    Returns None for unknown corpus or when scoping is disabled.
    """
    if not settings.enable_corpus_scoped_retrieval:
        return None

    key = (corpus or "").lower().strip()
    if key not in _CORPUS_DOCUMENT_TYPE:
        return None

    resolved_source = source_file or _CORPUS_DEFAULT_SOURCE_FILE.get(key)
    if resolved_source:
        return {"source_file": resolved_source}

    return {"document_type": _CORPUS_DOCUMENT_TYPE[key]}


def node_matches_filters(node: NodeWithScore, filters: dict[str, Any]) -> bool:
    meta = node.metadata or {}
    for key, expected in filters.items():
        if meta.get(key) != expected:
            return False
    return True


def filter_nodes_by_metadata(
    nodes: list[NodeWithScore],
    filters: dict[str, Any] | None,
) -> list[NodeWithScore]:
    if not filters:
        return nodes
    return [n for n in nodes if node_matches_filters(n, filters)]


class ScopeFilteredRetriever:
    """Post-filter retrieval results (especially BM25) to a metadata scope."""

    def __init__(self, inner: Any, filters: dict[str, Any]) -> None:
        self._inner = inner
        self._filters = filters

    def retrieve(self, query: Any) -> list[NodeWithScore]:
        nodes = self._inner.retrieve(query)
        return filter_nodes_by_metadata(nodes, self._filters)

    def retrieve_comprehensive(self, raw_query: str) -> list[NodeWithScore]:
        if hasattr(self._inner, "retrieve_comprehensive"):
            nodes = self._inner.retrieve_comprehensive(raw_query)
        else:
            nodes = self._inner.retrieve(raw_query)
        return filter_nodes_by_metadata(nodes, self._filters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)