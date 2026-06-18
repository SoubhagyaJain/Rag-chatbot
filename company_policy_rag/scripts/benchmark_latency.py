#!/usr/bin/env python3
"""Measure p50/p95 pipeline latency on golden-set questions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from src.config import settings
from src.evaluation import create_retriever, load_golden_dataset
from src.generation import generate_grounded_answer_with_trace
from src.indexing import configure_llama_index, index_exists, load_index
from src.prompts import get_generation_config_summary
from src.retriever import get_retrieval_config_summary
from src.timing import (
    begin_query_timing,
    clear_timing,
    get_current_timing,
    record_stage,
    summarize_ms,
)
from src.utils import logger, setup_logging


def run_timed_query(question: str, index) -> dict[str, float]:
    begin_query_timing()
    t0 = time.perf_counter()
    retriever = create_retriever(index)
    nodes = retriever.retrieve(question)
    generate_grounded_answer_with_trace(question, nodes, Settings.llm)
    record_stage("e2e", (time.perf_counter() - t0) * 1000)
    timing = get_current_timing()
    result = timing.as_dict() if timing else {}
    clear_timing()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark RAG pipeline latency")
    parser.add_argument("--iterations", type=int, default=3, help="Runs per question")
    parser.add_argument("--limit", type=int, default=0, help="Max golden cases (0 = all)")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True, help="Drop first iter per question")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "logs" / "latency_benchmark.json",
    )
    args = parser.parse_args(argv)

    setup_logging()
    if not index_exists():
        logger.error("No index — run scripts/index_documents.py first")
        return 1

    configure_llama_index()
    Settings.llm = Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        request_timeout=settings.llm_request_timeout,
    )
    index = load_index()
    cases = load_golden_dataset()
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    samples: list[dict[str, float]] = []
    for case in cases:
        for it in range(1, args.iterations + 1):
            logger.info("Timing %s iter %d/%d", case.id, it, args.iterations)
            try:
                row = run_timed_query(case.question, index)
                row["case_id"] = case.id
                row["iteration"] = it
                if args.warmup and it == 1:
                    row["discarded"] = True
                else:
                    row["discarded"] = False
                    samples.append(row)
            except Exception as exc:
                logger.error("Failed %s iter %d: %s", case.id, it, exc)
                return 1

    stages = [
        "query_rewrite_ms",
        "chroma_retrieve_ms",
        "rerank_filter_ms",
        "retrieve_total_ms",
        "generation_ms",
        "faithfulness_guard_ms",
        "e2e_ms",
    ]
    summary = {
        stage: summarize_ms([s[stage] for s in samples if stage in s])
        for stage in stages
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model,
        "retrieval_config": get_retrieval_config_summary(),
        "generation_config": get_generation_config_summary(),
        "sample_count": len(samples),
        "discarded_warmup_per_case": args.warmup,
        "iterations_per_question": args.iterations,
        "summary_ms": summary,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d samples)", args.output, len(samples))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())