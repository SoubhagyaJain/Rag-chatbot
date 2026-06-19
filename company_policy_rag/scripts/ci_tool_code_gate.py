#!/usr/bin/env python3
"""
CI tool/code generation gate — retrieval + judge relevancy on currency cases.

Usage:
    python scripts/ci_tool_code_gate.py
    python scripts/ci_tool_code_gate.py --retrieval-only

Exits 0 when currency tool/code cases meet retrieval floors and (unless --retrieval-only)
minimum relevancy on the tool-code subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_evaluation
from src.utils import logger, setup_logging

DATASET = PROJECT_ROOT / "data/eval/golden_subset_tool_code.json"
RETRIEVAL_METRICS = ("hit_rate", "context_precision", "context_recall")
RETRIEVAL_FLOORS = {
    "hit_rate": 1.0,
    "context_precision": 0.5,
    "context_recall": 0.8,
}
MIN_RELEVANCY = 0.8
CURRENCY_CASE_IDS = frozenset({"currency_tool_example", "tools_real_world"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI tool/code generation gate.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM judge; check retrieval metrics only.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="Tool/code golden subset path.",
    )
    parser.add_argument(
        "--min-relevancy",
        type=float,
        default=MIN_RELEVANCY,
        help="Minimum relevancy per currency case (full eval mode).",
    )
    return parser.parse_args(argv)


def _check_retrieval(aggregate: dict[str, float | None]) -> list[str]:
    failures: list[str] = []
    for metric, floor in RETRIEVAL_FLOORS.items():
        actual = aggregate.get(metric)
        if actual is None:
            failures.append(f"{metric}: missing value")
        elif actual < floor:
            failures.append(f"{metric}: {actual:.3f} < floor {floor:.3f}")
    return failures


def _check_relevancy(cases: list[dict], min_rel: float) -> list[str]:
    failures: list[str] = []
    for case in cases:
        case_id = case.get("id", "")
        if case_id not in CURRENCY_CASE_IDS:
            continue
        rel = case.get("answer_relevancy")
        if rel is None:
            failures.append(f"{case_id}: relevancy missing")
        elif rel < min_rel:
            failures.append(f"{case_id}: relevancy {rel:.3f} < floor {min_rel:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    setup_logging("ci_tool_code_gate")
    args = parse_args(argv)

    if not args.dataset.exists():
        logger.error("Tool/code dataset not found: %s", args.dataset)
        return 1

    try:
        run = run_evaluation(
            dataset_path=args.dataset,
            use_llm_judge=not args.retrieval_only,
            retrieval_only=args.retrieval_only,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("CI tool/code gate failed: %s", exc)
        return 1

    failures = _check_retrieval(run.aggregate)
    if not args.retrieval_only:
        failures.extend(_check_relevancy(run.cases, args.min_relevancy))

    logger.info(
        "Tool/code gate: hit=%.3f prec=%.3f rec=%.3f rel=%.3f (%d cases, %.1fs)",
        run.aggregate.get("hit_rate") or 0.0,
        run.aggregate.get("context_precision") or 0.0,
        run.aggregate.get("context_recall") or 0.0,
        run.aggregate.get("answer_relevancy") or 0.0,
        run.case_count,
        run.duration_seconds,
    )

    if failures:
        for msg in failures:
            logger.error("CI tool/code gate FAIL: %s", msg)
        return 1

    logger.info("CI tool/code gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())