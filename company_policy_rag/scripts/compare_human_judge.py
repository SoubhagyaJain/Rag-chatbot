#!/usr/bin/env python3
"""Compare human eval scores vs LLM judge on 5-case overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.human_judge_agreement import compare_human_llm, load_eval_run_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "human_eval_scores.json",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "human_eval_subset.json",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=PROJECT_ROOT / "logs" / "evaluation_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "logs" / "human_judge_agreement.json",
    )
    args = parser.parse_args(argv)

    human = json.loads(args.human.read_text(encoding="utf-8"))
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    run_id = human.get("run_id") or subset["run_id"]
    case_ids = subset["case_ids"]

    llm_cases = load_eval_run_cases(args.eval_results, run_id)
    report = compare_human_llm(human, llm_cases, case_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())