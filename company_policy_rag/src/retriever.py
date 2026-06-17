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

from src.citations import log_retrieval_stage
from src.config import settings
from src.indexing import load_index
from src.postprocessors import RelativeScoreThresholdPostprocessor
from src.query_processing import rewrite_query_for_retrieval
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


def get_final_top_k() -> int:
    """Chunks ultimately passed to generation after optional rerank + filtering."""
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
    if filters:
        kwargs["filters"] = filters

    base_retriever = idx.as_retriever(**kwargs)

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
        return _QueryRewritingRetriever(retriever)

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
        nodes = self._inner.retrieve(bundle)
        log_retrieval_stage("chroma_retrieved", nodes)
        filtered = apply_postprocessors(nodes, bundle, self._postprocessors)
        log_retrieval_stage("post_rerank_filter", filtered)
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
        if isinstance(query, QueryBundle):
            raw = query.query_str
            rewritten = preprocess_query(raw)
            bundle = QueryBundle(query_str=rewritten)
        else:
            bundle = QueryBundle(query_str=preprocess_query(query))
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
    }


def reset_reranker_cache() -> None:
    """Clear cached reranker instance (useful in tests)."""
    global _reranker, _reranker_loaded
    _reranker = None
    _reranker_loaded = False