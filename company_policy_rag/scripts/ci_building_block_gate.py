#!/usr/bin/env python3
"""
CI round-1 building-block gate — guardrails, planning, abstention (3 cases).

Usage:
    python scripts/ci_building_block_gate.py
    python scripts/ci_building_block_gate.py --retrieval-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_evaluation
from src.utils import logger, setup_logging

DATASET = PROJECT_ROOT / "data/eval/golden_subset_building_blocks_round1.json"
PRIORITY_CASES = frozenset(
    {"guardrails_block", "planning_block", "abstention_company_vacation"}
)
RETRIEVAL_FLOORS = {
    "hit_rate": 0.66,
    "context_precision": 0.3,
    "context_recall": 0.6,
}
MIN_RELEVANCY = 0.75


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI building-block round-1 gate.")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--min-relevancy", type=float, default=MIN_RELEVANCY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging("ci_building_block_gate")
    args = parse_args(argv)

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    try:
        run = run_evaluation(
            dataset_path=args.dataset,
            use_llm_judge=not args.retrieval_only,
            retrieval_only=args.retrieval_only,
        )
    except Exception as exc:
        logger.exception("CI building-block gate failed: %s", exc)
        return 1

    failures: list[str] = []
    for metric, floor in RETRIEVAL_FLOORS.items():
        actual = run.aggregate.get(metric)
        if actual is None or actual < floor:
            failures.append(f"{metric}: {actual} < floor {floor}")

    if not args.retrieval_only:
        for case in run.cases:
            if case["id"] not in PRIORITY_CASES:
                continue
            rel = case.get("answer_relevancy")
            if rel is None or rel < args.min_relevancy:
                failures.append(
                    f"{case['id']}: relevancy {rel} < floor {args.min_relevancy}"
                )

    logger.info(
        "Building-block gate: hit=%.3f rel=%.3f (%d cases)",
        run.aggregate.get("hit_rate") or 0.0,
        run.aggregate.get("answer_relevancy") or 0.0,
        run.case_count,
    )

    if failures:
        for msg in failures:
            logger.error("CI building-block gate FAIL: %s", msg)
        return 1

    logger.info("CI building-block gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())