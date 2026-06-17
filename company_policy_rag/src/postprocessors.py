"""
Custom node post-processors for retrieval precision.

Applied after cross-encoder reranking to drop marginal chunks whose scores
are far below the best match — a high-ROI fix when Context Precision is low
but Context Recall is already acceptable.
"""

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.config import settings
from src.utils import logger


class RelativeScoreThresholdPostprocessor(BaseNodePostprocessor):
    """
    Drop nodes scoring below `min_ratio` of the top reranker score.

    Cross-encoder scores are raw logits (not 0–1). Relative thresholding adapts
    per query: if the best match is weak, we keep fewer/noisy-free chunks; if
    strong, we keep more — without a fixed global cutoff.

    Example: top score 8.2, min_ratio=0.45 → keep nodes with score >= 3.69
    """

    min_ratio: float = Field(
        default=0.45,
        description="Minimum fraction of the top node's score required to keep a chunk.",
    )
    min_keep: int = Field(
        default=1,
        description="Always keep at least this many nodes (if any exist).",
    )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        scored = [n for n in nodes if n.score is not None]
        if not scored:
            return nodes

        top_score = max(n.score for n in scored if n.score is not None)
        if top_score <= 0:
            return nodes[: self.min_keep]

        threshold = top_score * self.min_ratio
        filtered = [n for n in nodes if n.score is not None and n.score >= threshold]

        if len(filtered) < self.min_keep:
            # Fall back to top min_keep by score rather than returning nothing
            filtered = sorted(
                [n for n in nodes if n.score is not None],
                key=lambda n: n.score or 0,
                reverse=True,
            )[: self.min_keep]

        dropped = len(nodes) - len(filtered)
        log_fn = logger.info if settings.enable_citation_pipeline_logging else logger.debug
        if dropped or settings.enable_citation_pipeline_logging:
            log_fn(
                "RelativeScoreThreshold: kept %d/%d chunks (threshold=%.2f, top=%.2f, ratio=%.2f)",
                len(filtered),
                len(nodes),
                threshold,
                top_score,
                self.min_ratio,
            )
        return filtered