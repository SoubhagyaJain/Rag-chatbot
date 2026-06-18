"""Pipeline stage timing for latency benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": float(len(samples)),
        "mean": round(sum(samples) / len(samples), 1),
        "p50": round(percentile(samples, 50), 1),
        "p95": round(percentile(samples, 95), 1),
    }


@dataclass
class PipelineTiming:
    """Per-query stage timings in milliseconds."""

    query_rewrite_ms: float = 0.0
    chroma_retrieve_ms: float = 0.0
    rerank_filter_ms: float = 0.0
    generation_ms: float = 0.0
    faithfulness_guard_ms: float = 0.0
    e2e_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieve_total_ms(self) -> float:
        return self.chroma_retrieve_ms + self.rerank_filter_ms

    def as_dict(self) -> dict[str, float]:
        return {
            "query_rewrite_ms": round(self.query_rewrite_ms, 1),
            "chroma_retrieve_ms": round(self.chroma_retrieve_ms, 1),
            "rerank_filter_ms": round(self.rerank_filter_ms, 1),
            "retrieve_total_ms": round(self.retrieve_total_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "faithfulness_guard_ms": round(self.faithfulness_guard_ms, 1),
            "e2e_ms": round(self.e2e_ms, 1),
        }


_current: PipelineTiming | None = None


def begin_query_timing() -> PipelineTiming:
    global _current
    _current = PipelineTiming()
    return _current


def get_current_timing() -> PipelineTiming | None:
    return _current


def record_stage(stage: str, elapsed_ms: float) -> None:
    if _current is None:
        return
    mapping = {
        "query_rewrite": "query_rewrite_ms",
        "chroma_retrieve": "chroma_retrieve_ms",
        "rerank_filter": "rerank_filter_ms",
        "generation": "generation_ms",
        "faithfulness_guard": "faithfulness_guard_ms",
        "e2e": "e2e_ms",
    }
    attr = mapping.get(stage)
    if attr:
        setattr(_current, attr, elapsed_ms)


def clear_timing() -> None:
    global _current
    _current = None