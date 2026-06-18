#!/usr/bin/env python3
"""Diagnose code validation for a single golden case or ad-hoc question."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llama_index.core import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.code_validation import (
    extract_code_lines,
    heuristic_code_grounded,
    should_validate_code,
)
from src.config import settings
from src.evaluation import load_golden_dataset
from src.generation import generate_grounded_answer_with_trace
from src.indexing import configure_llama_index, create_retriever, load_index
from src.prompts import format_nodes_for_prompt
from src.utils import setup_logging


def _load_case(case_id: str) -> dict:
    for path in (
        settings.eval_guidebook_dataset_path,
        settings.eval_dataset_path,
        PROJECT_ROOT / "data/eval/golden_subset_weak_guidebook.json",
    ):
        if not path.exists():
            continue
        for case in load_golden_dataset(path):
            if case.id == case_id:
                return {
                    "id": case.id,
                    "question": case.question,
                    "corpus": case.corpus,
                    "query_type": case.query_type,
                }
    raise ValueError(f"Case id not found: {case_id}")


def main(argv: list[str] | None = None) -> int:
    setup_logging("diagnose_code_validation")
    parser = argparse.ArgumentParser(description="Diagnose code validation for one case")
    parser.add_argument("--case-id", type=str, help="Golden case id (e.g. tools_real_world)")
    parser.add_argument("--question", type=str, help="Ad-hoc question (requires manual review)")
    parser.add_argument("--dry-run", action="store_true", help="Retrieval + trace only, print code lines")
    args = parser.parse_args(argv)

    if not args.case_id and not args.question:
        parser.error("Provide --case-id or --question")

    configure_llama_index()
    index = load_index()
    retriever = create_retriever(index)

    if args.case_id:
        case = _load_case(args.case_id)
        question = case["question"]
        print(f"Case: {case['id']} ({case['query_type']})")
    else:
        question = args.question or ""
        case = {"id": "adhoc", "query_type": "?"}
        print("Case: adhoc")

    print(f"Question: {question}")
    print("-" * 72)

    nodes = retriever.retrieve(question)
    print(f"Retrieved chunks: {len(nodes)}")

    if args.dry_run:
        context = format_nodes_for_prompt(nodes)
        print(f"Context length: {len(context)} chars")
        return 0

    trace = generate_grounded_answer_with_trace(question, nodes, Settings.llm)
    answer = trace.final_answer
    pre = trace.post_guard_answer

    print(f"Trigger would fire: {should_validate_code(pre, nodes)}")
    print(f"Code lines in post-guard answer: {extract_code_lines(pre)}")

    context = format_nodes_for_prompt(nodes)
    h = heuristic_code_grounded(pre, context)
    print(f"Heuristic on post-guard: passed={h.passed} failed={h.failed[:3]}")

    print("-" * 72)
    print(f"code_validation.triggered: {trace.code_validation.triggered}")
    print(f"code_validation.passed: {trace.code_validation.passed}")
    print(f"code_validation.method: {trace.code_validation.validation_method}")
    print(f"code_validation.explanation: {trace.code_validation.explanation}")
    if trace.code_validation.failed_lines:
        print(f"code_validation.failed_lines: {trace.code_validation.failed_lines[:5]}")
    print(f"fallback_reason: {trace.fallback_reason}")
    print("-" * 72)
    print("POST-GUARD (first 400 chars):")
    print(pre[:400])
    print("-" * 72)
    print("FINAL (first 400 chars):")
    print(answer[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())