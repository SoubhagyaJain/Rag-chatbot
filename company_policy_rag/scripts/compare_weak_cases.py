#!/usr/bin/env python3
"""Compare weak-case metrics between two eval runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEAK_IDS = [
    "tools_real_world",
    "currency_tool_example",
    "custom_tools",
    "code_available_links",
    "code_check_this_out",
    "six_building_blocks",
    "design_patterns_popular",
    "memory_short_long",
    "enumeration_subagents",
    "agent_building_blocks_count",
]

TARGET_RETRIEVAL_IDS = [
    "currency_tool_example",
    "tools_real_world",
    "code_available_links",
    "code_check_this_out",
]

RETRIEVAL_METRICS = ("hit_rate", "context_precision", "context_recall")
FULL_METRICS = ("hit_rate", "context_precision", "context_recall", "answer_relevancy", "faithfulness")


def _cases_by_id(path: Path, run_id: str | None, ids: list[str]) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No runs in {path}")
    if run_id:
        run = next(r for r in runs if r.get("run_id") == run_id)
    else:
        run = runs[-1]
    return {c["id"]: c for c in run["cases"] if c["id"] in ids}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare eval runs for weak guidebook cases.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--post-run", default=None)
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Compare only the four code/currency retrieval target cases",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Show retrieval metrics only (hit, precision, recall)",
    )
    args = parser.parse_args(argv)

    case_ids = TARGET_RETRIEVAL_IDS if args.target_only else WEAK_IDS
    metrics = RETRIEVAL_METRICS if args.retrieval_only else FULL_METRICS

    baseline = _cases_by_id(args.baseline, args.baseline_run, case_ids)
    post = _cases_by_id(args.post, args.post_run, case_ids)

    print(f"{'id':<30} {'metric':<18} {'baseline':>8} {'post':>8} {'delta':>8}")
    print("-" * 78)
    for cid in case_ids:
        b, p = baseline[cid], post[cid]
        for key in metrics:
            bv, pv = b.get(key), p.get(key)
            if bv is None and pv is None:
                continue
            delta = (pv - bv) if bv is not None and pv is not None else None
            ds = f"{delta:+.2f}" if delta is not None else "—"
            b_str = f"{bv:.2f}" if bv is not None else "—"
            p_str = f"{pv:.2f}" if pv is not None else "—"
            print(f"{cid:<30} {key:<18} {b_str:>8} {p_str:>8} {ds:>8}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())