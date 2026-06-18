#!/usr/bin/env python3
"""Compare weak-case metrics baseline vs post-reindex subset run."""

from __future__ import annotations

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


def _cases_by_id(path: Path, run_id: str) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    run = next(r for r in data["runs"] if r["run_id"] == run_id)
    return {c["id"]: c for c in run["cases"] if c["id"] in WEAK_IDS}


def main() -> int:
    baseline = _cases_by_id(
        PROJECT_ROOT / "logs/evaluation_guidebook.json", "20260618_132316"
    )
    post = _cases_by_id(
        PROJECT_ROOT / "logs/evaluation_subset_weak.json", "20260618_140509"
    )
    print(f"{'id':<30} {'metric':<12} {'baseline':>8} {'post':>8} {'delta':>8}")
    print("-" * 72)
    for cid in WEAK_IDS:
        b, p = baseline[cid], post[cid]
        for key in ("hit_rate", "answer_relevancy", "faithfulness", "context_precision"):
            bv, pv = b.get(key), p.get(key)
            delta = (pv - bv) if bv is not None and pv is not None else None
            ds = f"{delta:+.2f}" if delta is not None else "—"
            print(f"{cid:<30} {key:<12} {bv:>8.2f} {pv:>8.2f} {ds:>8}")
        print(
            f"{cid:<30} {'fallback':<12} {str(b.get('fallback_reason')):>8} "
            f"{str(p.get('fallback_reason')):>8}"
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())