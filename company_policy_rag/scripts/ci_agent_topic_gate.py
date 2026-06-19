#!/usr/bin/env python3
"""
CI round-2 agent-topic gate — building blocks + manager/RAG/memory/custom tools.

Usage:
    python scripts/ci_agent_topic_gate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_evaluation
from src.utils import logger, setup_logging

DATASET = PROJECT_ROOT / "data/eval/golden_subset_agent_round2.json"
PRIORITY_CASES = frozenset(
    {
        "guardrails_block",
        "planning_block",
        "abstention_company_vacation",
        "memory_block",
        "manager_agent",
        "rag_in_agent",
        "custom_tools",
    }
)
MIN_RELEVANCY = 0.75


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI agent-topic round-2 gate.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--min-relevancy", type=float, default=MIN_RELEVANCY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging("ci_agent_topic_gate")
    args = parse_args(argv)

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    try:
        run = run_evaluation(
            dataset_path=args.dataset,
            use_llm_judge=True,
            retrieval_only=False,
        )
    except Exception as exc:
        logger.exception("CI agent-topic gate failed: %s", exc)
        return 1

    failures: list[str] = []
    for case in run.cases:
        if case["id"] not in PRIORITY_CASES:
            continue
        rel = case.get("answer_relevancy")
        if rel is None or rel < args.min_relevancy:
            failures.append(
                f"{case['id']}: relevancy {rel} < floor {args.min_relevancy}"
            )

    logger.info(
        "Agent-topic gate: rel=%.3f (%d cases)",
        run.aggregate.get("answer_relevancy") or 0.0,
        run.case_count,
    )

    if failures:
        for msg in failures:
            logger.error("CI agent-topic gate FAIL: %s", msg)
        return 1

    logger.info("CI agent-topic gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())