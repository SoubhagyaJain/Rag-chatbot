"""
Retrieval pipeline with optional cross-encoder reranking.

Production-rag precision pipeline (low Context Precision fix):
  1. Optional LLM query rewrite → keyword-dense search query
  2. Over-retrieve from Chroma (top 25–30) — recall pool for reranker
  3. Cross-encoder rerank (bge-reranker-large) → top 5–7
  4. Relative score threshold → drop marginal chunks below 45% of top score

Tuning guide (policy/legal docs):
  - Low precision, good recall → lower RERANKER_TOP_N, enable score threshold, use large reranker
  - Low recall → raise RETRIEVAL_CANDIDATE_K, loosen RERANK_MIN_SCORE_RATIO
"""

from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.bm25_index import get_bm25_corpus_size
from src.retrieval_scope import ScopeFilteredRetriever
from src.citations import log_retrieval_stage
from src.config import settings
from src.hybrid_retrieval import HybridRetriever, bm25_nodes_for_query, reciprocal_rank_fusion
from src.indexing import load_index
from src.parent_retrieval import expand_to_parent_documents
from src.postprocessors import RelativeScoreThresholdPostprocessor
from src.query_processing import (
    build_multi_retrieval_queries,
    is_comprehensive_list_query,
    rewrite_query_for_retrieval,
)
from src.timing import get_current_timing, record_stage
from src.utils import timer
from src.utils import logger

# Lazy-loaded singleton — cross-encoder load is ~1–3 s; avoid reloading per query
_reranker: BaseNodePostprocessor | None = None
_reranker_loaded: bool = False

# Shown in logs when reranker deps are missing (see README for CPU/GPU commands)
RERANKER_INSTALL_CPU = (
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && "
    "pip install sentence-transformers llama-index-postprocessor-sbert-rerank"
)
RERANKER_INSTALL_GPU_CUDA124 = (
    "pip install torch --index-url https://download.pytorch.org/whl/cu124 && "
    "pip install sentence-transformers llama-index-postprocessor-sbert-rerank"
)


def get_initial_top_k() -> int:
    """
    Chroma retrieval depth before reranking.

    When reranker is on, over-retrieve then trim; otherwise use similarity_top_k.
    """
    if settings.enable_reranker:
        return settings.retrieval_candidate_k
    return settings.similarity_top_k


def get_final_top_k(*, comprehensive: bool = False) -> int:
    """Chunks ultimately passed to generation after optional rerank + filtering."""
    if comprehensive and settings.enable_comprehensive_retrieval:
        return settings.comprehensive_reranker_top_n
    if settings.enable_reranker:
        return settings.reranker_top_n
    return settings.similarity_top_k

def get_reranker_install_hints() -> dict[str, str]:
    """Copy-paste install commands for reranker dependencies (CPU + GPU)."""
    return {
        "cpu": RERANKER_INSTALL_CPU,
        "gpu_cuda124": RERANKER_INSTALL_GPU_CUDA124,
    }


def _check_reranker_dependencies() -> list[str]:
    """Return names of missing packages required by SentenceTransformerRerank."""
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")
    try:
        import llama_index.postprocessor.sbert_rerank  # noqa: F401
    except ImportError:
        missing.append("llama-index-postprocessor-sbert-rerank")
    return missing


def _resolve_reranker_device(requested: str) -> str:
    """
    Resolve RERANKER_DEVICE.

    Supports `cpu`, `cuda`, `cuda:0`, or `auto` (cuda when available, else cpu).
    Falls back to cpu with a warning if CUDA is requested but unavailable.
    """
    device = (requested or "cpu").strip().lower()
    if device in ("auto", ""):
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
        return device

    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "RERANKER_DEVICE=%s but CUDA is not available — using cpu",
                    requested,
                )
                return "cpu"
        except ImportError:
            logger.warning(
                "RERANKER_DEVICE=%s but torch is not installed — using cpu",
                requested,
            )
            return "cpu"
    return device


def _log_reranker_fallback(reason: str, *, detail: str = "") -> None:
    """Emit actionable guidance when reranker cannot load."""
    hints = get_reranker_install_hints()
    logger.error(
        "Reranker unavailable (%s)%s — falling back to vector-only retrieval. "
        "Context Precision will be lower until deps are installed.",
        reason,
        f": {detail}" if detail else "",
    )
    logger.error("Install (CPU):  %s", hints["cpu"])
    logger.error("Install (GPU):  %s", hints["gpu_cuda124"])


def get_reranker() -> BaseNodePostprocessor | None:
    """
    Return the configured cross-encoder reranker, or None if disabled/unavailable.

    Uses SentenceTransformerRerank from llama_index.postprocessor.sbert_rerank.
    Requires: torch, sentence-transformers, llama-index-postprocessor-sbert-rerank.
    """
    global _reranker, _reranker_loaded

    if not settings.enable_reranker:
        logger.debug("Reranker disabled (ENABLE_RERANKER=false)")
        return None

    if _reranker_loaded:
        return _reranker

    missing = _check_reranker_dependencies()
    if missing:
        _log_reranker_fallback(
            "missing dependencies",
            detail=", ".join(missing),
        )
        _reranker = None
        _reranker_loaded = True
        return None

    device = _resolve_reranker_device(settings.reranker_device)

    try:
        from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

        _reranker = SentenceTransformerRerank(
            model=settings.reranker_model,
            top_n=settings.reranker_top_n,
            device=device,
        )

        logger.info(
            "Reranker loaded successfully | model=%s | candidates=%d → top_n=%d | "
            "device=%s (requested=%s)",
            settings.reranker_model,
            settings.retrieval_candidate_k,
            settings.reranker_top_n,
            device,
            settings.reranker_device,
        )
    except ImportError as exc:
        _log_reranker_fallback("import error", detail=str(exc))
        _reranker = None
    except Exception as exc:
        _log_reranker_fallback("initialization error", detail=str(exc))
        _reranker = None

    _reranker_loaded = True
    return _reranker


def get_node_postprocessors() -> list[BaseNodePostprocessor]:
    """
    Post-processors applied after Chroma retrieval, in order:
      1. Cross-encoder rerank
      2. Relative score threshold (precision filter)
    """
    processors: list[BaseNodePostprocessor] = []

    reranker = get_reranker()
    if reranker is not None:
        processors.append(reranker)
        if settings.enable_rerank_score_filter and settings.rerank_min_score_ratio > 0:
            processors.append(
                RelativeScoreThresholdPostprocessor(
                    min_ratio=settings.rerank_min_score_ratio,
                    min_keep=settings.rerank_min_keep,
                )
            )
        logger.debug(
            "Retrieval post-processors: rerank + %s",
            "score_filter" if len(processors) > 1 else "rerank_only",
        )
    elif settings.enable_reranker:
        logger.warning(
            "Reranker enabled but not loaded — retrieval uses vector-only top_k=%d",
            get_initial_top_k(),
        )

    return processors


def preprocess_query(query: str) -> str:
    """Apply query rewrite before vector search (shared by retriever + eval)."""
    return rewrite_query_for_retrieval(query)


def _to_query_bundle(query: str | QueryBundle) -> QueryBundle:
    if isinstance(query, QueryBundle):
        return query
    return QueryBundle(query_str=query)


def apply_postprocessors(
    nodes: list[NodeWithScore],
    query_bundle: QueryBundle,
    postprocessors: list[BaseNodePostprocessor],
) -> list[NodeWithScore]:
    """Run rerank + score filter on retrieved nodes (VectorIndexRetriever does not)."""
    for postprocessor in postprocessors:
        nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
    return nodes


def _diversify_comprehensive_nodes(
    nodes: list[NodeWithScore],
    *,
    max_per_section: int = 2,
) -> list[NodeWithScore]:
    """Cap chunks per section so list questions get heterogeneous rerank input."""
    buckets: dict[str, list[NodeWithScore]] = {}
    for node in nodes:
        meta = node.metadata or {}
        section = str(meta.get("section_path") or meta.get("section_title") or "unknown")
        buckets.setdefault(section, []).append(node)

    diversified: list[NodeWithScore] = []
    for section_nodes in buckets.values():
        section_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
        diversified.extend(section_nodes[:max_per_section])
    diversified.sort(key=lambda n: n.score or 0.0, reverse=True)
    return diversified


def build_retriever(
    index: VectorStoreIndex | None = None,
    *,
    filters: dict[str, Any] | None = None,
    apply_query_rewrite: bool | None = None,
):
    """
    Build a Chroma retriever with rewrite → retrieve → rerank → score filter.

    Metadata filters pass through to Chroma unchanged.
    """
    idx = index or load_index()
    kwargs: dict[str, Any] = {"similarity_top_k": get_initial_top_k()}

    base_retriever = idx.as_retriever(**kwargs)
    if settings.enable_hybrid_bm25:
        base_retriever = HybridRetriever(base_retriever)

    postprocessors = get_node_postprocessors()
    retriever: Any = base_retriever
    if postprocessors:
        retriever = _PostprocessingRetriever(base_retriever, postprocessors)

    should_rewrite = (
        apply_query_rewrite
        if apply_query_rewrite is not None
        else settings.enable_query_rewrite
    )

    if should_rewrite:
        retriever = _QueryRewritingRetriever(retriever)

    if filters:
        retriever = ScopeFilteredRetriever(retriever, filters)

    return retriever


class _PostprocessingRetriever:
    """
    Apply rerank + score filter after Chroma retrieval.

    LlamaIndex VectorIndexRetriever ignores node_postprocessors; only
    RetrieverQueryEngine applies them — this wrapper fixes direct .retrieve() calls.
    """

    def __init__(
        self,
        inner_retriever: Any,
        postprocessors: list[BaseNodePostprocessor],
    ) -> None:
        self._inner = inner_retriever
        self._postprocessors = postprocessors

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        bundle = _to_query_bundle(query)
        return self._retrieve_bundle(bundle)

    def retrieve_comprehensive(self, raw_query: str) -> list[NodeWithScore]:
        """Multi-query Chroma retrieval for list/enumeration questions."""
        sub_queries = build_multi_retrieval_queries(
            raw_query,
            max_queries=settings.comprehensive_max_subqueries,
        )
        merged: dict[str, NodeWithScore] = {}
        with timer("chroma_retrieve") as t_chroma:
            for sub_query in sub_queries:
                rewritten = preprocess_query(sub_query)
                bundle = QueryBundle(query_str=rewritten)
                dense_nodes: list[NodeWithScore] = []
                if hasattr(self._inner, "_dense"):
                    dense_nodes = self._inner._dense.retrieve(bundle)
                else:
                    dense_nodes = self._inner.retrieve(bundle)

                if settings.enable_hybrid_bm25:
                    bm25_hits = bm25_nodes_for_query(rewritten)
                    sub_merged = reciprocal_rank_fusion([dense_nodes, bm25_hits])
                else:
                    sub_merged = dense_nodes

                for node in sub_merged:
                    node_id = node.node.node_id
                    score = node.score or 0.0
                    existing = merged.get(node_id)
                    if existing is None or score > (existing.score or 0.0):
                        merged[node_id] = node
        if get_current_timing() is not None:
            record_stage("chroma_retrieve", t_chroma["elapsed_ms"])

        nodes = _diversify_comprehensive_nodes(list(merged.values()))
        log_retrieval_stage("chroma_retrieved_comprehensive", nodes)
        primary = QueryBundle(query_str=preprocess_query(raw_query))
        reranker = get_reranker()
        original_top_n: int | None = None
        if reranker is not None and hasattr(reranker, "top_n"):
            original_top_n = reranker.top_n
            reranker.top_n = max(
                settings.comprehensive_reranker_top_n,
                settings.reranker_top_n,
            )
        with timer("rerank_filter") as t_rerank:
            try:
                filtered = apply_postprocessors(nodes, primary, self._postprocessors)
            finally:
                if original_top_n is not None and reranker is not None:
                    reranker.top_n = original_top_n
            if not filtered and nodes:
                rerank_only = [
                    p
                    for p in self._postprocessors
                    if not isinstance(p, RelativeScoreThresholdPostprocessor)
                ]
                logger.warning(
                    "Comprehensive retrieval: score filter removed all %d merged nodes; "
                    "retrying rerank-only",
                    len(nodes),
                )
                filtered = apply_postprocessors(nodes, primary, rerank_only)
            if not filtered and nodes:
                filtered = sorted(
                    nodes,
                    key=lambda n: n.score or 0.0,
                    reverse=True,
                )
        if get_current_timing() is not None:
            record_stage("rerank_filter", t_rerank["elapsed_ms"])
        top_n = get_final_top_k(comprehensive=True)
        filtered = filtered[:top_n]
        log_retrieval_stage("post_rerank_filter", filtered)
        filtered = expand_to_parent_documents(filtered)
        log_retrieval_stage("post_parent_expand", filtered)
        return filtered

    def _retrieve_bundle(self, bundle: QueryBundle) -> list[NodeWithScore]:
        with timer("chroma_retrieve") as t_chroma:
            nodes = self._inner.retrieve(bundle)
        if get_current_timing() is not None:
            record_stage("chroma_retrieve", t_chroma["elapsed_ms"])
        log_retrieval_stage("chroma_retrieved", nodes)
        with timer("rerank_filter") as t_rerank:
            filtered = apply_postprocessors(nodes, bundle, self._postprocessors)
        if get_current_timing() is not None:
            record_stage("rerank_filter", t_rerank["elapsed_ms"])
        log_retrieval_stage("post_rerank_filter", filtered)
        filtered = expand_to_parent_documents(filtered)
        log_retrieval_stage("post_parent_expand", filtered)
        return filtered

    async def aretrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        bundle = _to_query_bundle(query)
        if hasattr(self._inner, "aretrieve"):
            nodes = await self._inner.aretrieve(bundle)
        else:
            nodes = self._inner.retrieve(bundle)
        result = nodes
        for postprocessor in self._postprocessors:
            if hasattr(postprocessor, "apostprocess_nodes"):
                result = await postprocessor.apostprocess_nodes(
                    result, query_bundle=bundle
                )
            else:
                result = postprocessor.postprocess_nodes(result, query_bundle=bundle)
        return result


class _QueryRewritingRetriever:
    """
    Thin wrapper: rewrite query → delegate to inner retriever.

    Keeps rewrite logic out of Chainlit / agent / eval call sites.
    """

    def __init__(self, inner_retriever: Any) -> None:
        self._inner = inner_retriever

    def retrieve(self, query: str | QueryBundle) -> Any:
        raw = query.query_str if isinstance(query, QueryBundle) else query
        if (
            settings.enable_comprehensive_retrieval
            and is_comprehensive_list_query(raw)
            and hasattr(self._inner, "retrieve_comprehensive")
        ):
            logger.info("Comprehensive list retrieval enabled for query")
            return self._inner.retrieve_comprehensive(raw)

        if isinstance(query, QueryBundle):
            rewritten = preprocess_query(raw)
            bundle = QueryBundle(query_str=rewritten)
        else:
            bundle = QueryBundle(query_str=preprocess_query(raw))
        return self._inner.retrieve(bundle)

    # LlamaIndex QueryEngine may call aretrieve
    async def aretrieve(self, query: str | QueryBundle) -> Any:
        if hasattr(self._inner, "aretrieve"):
            if isinstance(query, QueryBundle):
                raw = query.query_str
                bundle = QueryBundle(query_str=preprocess_query(raw))
            else:
                bundle = QueryBundle(query_str=preprocess_query(query))
            return await self._inner.aretrieve(bundle)
        return self.retrieve(query)


def build_query_engine(
    index: VectorStoreIndex | None = None,
    *,
    filters: dict[str, Any] | None = None,
):
    """
    Query engine: rewrite → retrieve → rerank → grounded synthesis → faithfulness guard.
    """
    from src.generation import build_grounded_query_engine

    return build_grounded_query_engine(index, filters=filters)


def get_retrieval_config_summary() -> dict[str, Any]:
    """Snapshot of retrieval settings for eval logs and debugging."""
    return {
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "reranker_top_n": settings.reranker_top_n,
        "reranker_model": settings.reranker_model,
        "enable_reranker": settings.enable_reranker,
        "enable_query_rewrite": settings.enable_query_rewrite,
        "rerank_min_score_ratio": settings.rerank_min_score_ratio,
        "enable_rerank_score_filter": settings.enable_rerank_score_filter,
        "enable_hybrid_bm25": settings.enable_hybrid_bm25,
        "bm25_top_k": settings.bm25_top_k,
        "hybrid_rrf_k": settings.hybrid_rrf_k,
        "bm25_corpus_size": get_bm25_corpus_size(),
        "enable_parent_document_retrieval": settings.enable_parent_document_retrieval,
        "enable_corpus_scoped_retrieval": settings.enable_corpus_scoped_retrieval,
    }


def reset_reranker_cache() -> None:
    """Clear cached reranker instance (useful in tests)."""
    global _reranker, _reranker_loaded
    _reranker = None
    _reranker_loaded = False