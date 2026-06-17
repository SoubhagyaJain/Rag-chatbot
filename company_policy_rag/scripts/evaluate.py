#!/usr/bin/env python3
"""
Run golden-set evaluation against the Chroma-backed RAG index.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --max-samples 5
    python scripts/evaluate.py --no-judge          # retrieval metrics only (faster)
    python scripts/evaluate.py --dataset data/eval/golden_dataset.json

Prerequisites:
    1. Index documents: python scripts/index_documents.py
    2. Ollama running with qwen2.5:7b (and eval model if different)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    format_results_table,
    run_evaluation,
    save_eval_results,
)
from src.utils import logger, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG quality against the golden dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to golden dataset JSON (default: data/eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of eval cases (for quick smoke tests)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge (retrieval metrics only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override results output path (default: logs/evaluation_results.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging("evaluate")
    args = parse_args(argv)

    try:
        run = run_evaluation(
            dataset_path=args.dataset,
            max_samples=args.max_samples,
            use_llm_judge=not args.no_judge,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 1

    table = format_results_table(run)
    print("\n" + table)

    save_eval_results(run, path=args.output)

    # Exit non-zero if primary metrics fall below thresholds (optional guardrail)
    agg = run.aggregate
    if agg.get("faithfulness") is not None and agg["faithfulness"] < 0.5:
        logger.warning("Faithfulness below 0.5 — review generation prompts or retrieval.")
    if agg.get("context_precision") is not None and agg["context_precision"] < 0.3:
        logger.warning("Context precision below 0.3 — review chunking or embeddings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())