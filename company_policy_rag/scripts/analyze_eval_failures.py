#!/usr/bin/env python3
"""Analyze failure modes from the latest (or named) eval run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

FAITH_THRESHOLD = 0.80
RELEVANCY_THRESHOLD = 0.75
PRECISION_THRESHOLD = 0.50


def _load_run(path: Path, run_id: str | None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No runs in {path}")
    if run_id:
        for run in reversed(runs):
            if run.get("run_id") == run_id:
                return run
        raise ValueError(f"Run id {run_id} not found")
    return runs[-1]


def _classify_failures(cases: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list) 

    for case in cases:
        cid = case["id"]
        faith = case.get("faithfulness")
        rel = case.get("answer_relevancy")
        prec = case.get("context_precision")
        hit = case.get("hit_rate", 0.0)
        fallback = case.get("fallback_reason", "none")
        guard = case.get("guard_modified", False)
        code_trig = case.get("code_validation_triggered", False)
        code_pass = case.get("code_validation_passed")

        if hit < 1.0:
            buckets["retrieval_miss"].append(case)
        if prec is not None and prec < PRECISION_THRESHOLD:
            buckets["low_context_precision"].append(case)
        if faith is not None and faith < FAITH_THRESHOLD:
            buckets["low_faithfulness"].append(case)
        if rel is not None and rel < RELEVANCY_THRESHOLD:
            buckets["low_relevancy"].append(case)
        if guard:
            buckets["faithfulness_guard_abstention"].append(case)
        if fallback == "code_validation":
            buckets["code_validation_fallback"].append(case)
        if code_trig and code_pass is False:
            buckets["code_validation_failed"].append(case)
        if case.get("category") == "edge_case" and rel is not None and rel < RELEVANCY_THRESHOLD:
            buckets["abstention_failure"].append(case)

    return dict(buckets)


def _print_summary(run: dict) -> None:
    agg = run.get("aggregate", {})
    print("=" * 72)
    print(f"Run ID:      {run.get('run_id')}")
    print(f"Timestamp:   {run.get('timestamp')}")
    print(f"Cases:       {run.get('case_count')}")
    print(f"Duration:    {run.get('duration_seconds')}s")
    print(f"Model:       {run.get('model')}")
    print("-" * 72)
    for key in (
        "hit_rate",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "code_validation_pass_rate",
        "low_confidence_fallback_rate",
    ):
        val = agg.get(key)
        if val is not None:
            print(f"  {key:<30} {val:.3f}")
    by_qt = agg.get("by_query_type", {})
    if by_qt:
        print("-" * 72)
        print("By query_type:")
        for qt, metrics in sorted(by_qt.items()):
            faith = metrics.get("faithfulness")
            rel = metrics.get("answer_relevancy")
            faith_s = f"{faith:.3f}" if faith is not None else "—"
            rel_s = f"{rel:.3f}" if rel is not None else "—"
            print(f"  {qt:<20} faith={faith_s}  relevancy={rel_s}  n={_count_type(run, qt)}")
    print("=" * 72)


def _count_type(run: dict, query_type: str) -> int:
    return sum(1 for c in run.get("cases", []) if c.get("query_type") == query_type)


def _print_bucket(name: str, cases: list[dict]) -> None:
    if not cases:
        return
    print(f"\n## {name} ({len(cases)} cases)")
    for c in cases:
        faith = c.get("faithfulness")
        rel = c.get("answer_relevancy")
        prec = c.get("context_precision")
        faith_s = f"{faith:.2f}" if faith is not None else "—"
        rel_s = f"{rel:.2f}" if rel is not None else "—"
        prec_s = f"{prec:.2f}" if prec is not None else "—"
        print(f"  - {c['id']}")
        print(f"    Q: {c['question'][:90]}{'…' if len(c['question']) > 90 else ''}")
        print(f"    hit={c.get('hit_rate', 0):.0f} prec={prec_s} faith={faith_s} rel={rel_s} "
              f"fallback={c.get('fallback_reason', 'none')} guard={c.get('guard_modified', False)}")
        notes = c.get("judge_notes", {})
        if notes.get("faithfulness"):
            print(f"    faith_note: {notes['faithfulness'][:120]}")
        if notes.get("relevancy"):
            print(f"    rel_note: {notes['relevancy'][:120]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze eval failure modes")
    parser.add_argument("--results", type=Path, default=settings.eval_results_path)
    parser.add_argument("--run-id", type=str, default=None, help="Specific run_id (default: latest)")
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"No results file: {args.results}", file=sys.stderr)
        return 1

    run = _load_run(args.results, args.run_id)
    cases = run.get("cases", [])
    buckets = _classify_failures(cases)

    _print_summary(run)
    for bucket in (
        "retrieval_miss",
        "low_context_precision",
        "low_faithfulness",
        "low_relevancy",
        "abstention_failure",
        "faithfulness_guard_abstention",
        "code_validation_fallback",
        "code_validation_failed",
    ):
        _print_bucket(bucket, buckets.get(bucket, []))

    worst = sorted(
        [c for c in cases if c.get("answer_relevancy") is not None],
        key=lambda c: (c.get("faithfulness") or 1.0, c.get("answer_relevancy") or 1.0),
    )[:5]
    if worst:
        print("\n## Worst 5 cases (lowest faith + relevancy)")
        for c in worst:
            print(
                f"  - {c['id']}: faith={c.get('faithfulness')} rel={c.get('answer_relevancy')} "
                f"| {c['question'][:70]}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())