#!/usr/bin/env python3
"""
CI retrieval smoke gate — runs --retrieval-only eval and fails on metric regression.

Usage:
    python scripts/ci_eval_gate.py
    python scripts/ci_eval_gate.py --write-baseline   # refresh ci_smoke_baseline.json locally

Exits 0 when hit_rate, context_precision, and context_recall meet CI_SMOKE_MIN_* floors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.evaluation import run_evaluation
from src.utils import logger, setup_logging

RETRIEVAL_METRICS = ("hit_rate", "context_precision", "context_recall")


def _thresholds() -> dict[str, float]:
    return {
        "hit_rate": settings.ci_smoke_min_hit_rate,
        "context_precision": settings.ci_smoke_min_context_precision,
        "context_recall": settings.ci_smoke_min_context_recall,
    }


def _check_metrics(aggregate: dict[str, float | None], floors: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric in RETRIEVAL_METRICS:
        actual = aggregate.get(metric)
        floor = floors[metric]
        if actual is None:
            failures.append(f"{metric}: missing value")
        elif actual < floor:
            failures.append(f"{metric}: {actual:.3f} < floor {floor:.3f}")
    return failures


def _write_baseline(run_aggregate: dict[str, float | None], *, run_id: str, case_count: int) -> Path:
    payload = {
        "version": "1.0",
        "description": "Retrieval-only CI smoke baseline (golden_subset_ci_smoke.json)",
        "run_id": run_id,
        "case_count": case_count,
        "aggregate": {m: run_aggregate.get(m) for m in RETRIEVAL_METRICS},
        "thresholds": _thresholds(),
    }
    out_path = settings.ci_smoke_baseline_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI retrieval smoke gate.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write ci_smoke_baseline.json from this run (local maintainer use).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override smoke dataset path (default: CI_SMOKE_DATASET_PATH).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging("ci_eval_gate")
    args = parse_args(argv)
    dataset_path = args.dataset or settings.ci_smoke_dataset_path

    if not dataset_path.exists():
        logger.error("CI smoke dataset not found: %s", dataset_path)
        return 1

    try:
        run = run_evaluation(
            dataset_path=dataset_path,
            use_llm_judge=False,
            retrieval_only=True,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("CI eval gate failed: %s", exc)
        return 1

    floors = _thresholds()
    failures = _check_metrics(run.aggregate, floors)

    logger.info(
        "CI smoke aggregate: hit=%.3f prec=%.3f rec=%.3f (%d cases, %.1fs)",
        run.aggregate.get("hit_rate") or 0.0,
        run.aggregate.get("context_precision") or 0.0,
        run.aggregate.get("context_recall") or 0.0,
        run.case_count,
        run.duration_seconds,
    )

    if args.write_baseline:
        path = _write_baseline(
            run.aggregate,
            run_id=run.run_id,
            case_count=run.case_count,
        )
        logger.info("Wrote baseline to %s", path)

    if failures:
        for msg in failures:
            logger.error("CI gate FAIL: %s", msg)
        return 1

    logger.info("CI gate PASS — all retrieval floors met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())