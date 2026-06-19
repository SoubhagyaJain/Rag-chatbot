#!/usr/bin/env python3
"""Print top-k retrieved chunks for a golden case ID (retrieval diagnostics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import load_golden_dataset
from src.indexing import configure_llama_index, create_retriever, index_exists, load_index
from src.retrieval_scope import corpus_retrieval_filters


def _preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug retrieval for a golden case.")
    parser.add_argument("case_id", help="Golden case id (e.g. currency_tool_example)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/eval/golden_subset_weak_guidebook.json",
    )
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)

    if not index_exists():
        print("Index missing — run scripts/index_documents.py first.", file=sys.stderr)
        return 1

    cases = load_golden_dataset(args.dataset)
    case = next((c for c in cases if c.id == args.case_id), None)
    if case is None:
        print(f"Case {args.case_id!r} not found in {args.dataset}", file=sys.stderr)
        return 1

    configure_llama_index()
    index = load_index()
    filters = corpus_retrieval_filters(case.corpus, source_file=case.source_file or None)
    retriever = create_retriever(index, filters=filters)
    nodes = retriever.retrieve(case.question)[: args.top_k]

    print(f"Case: {case.id}")
    print(f"Question: {case.question}")
    print(f"Relevant sections: {case.relevant_sections}")
    print(f"Retrieved: {len(nodes)} (showing up to {args.top_k})")
    print("-" * 72)
    for i, nws in enumerate(nodes, start=1):
        meta = nws.node.metadata or {}
        print(
            f"[{i}] score={nws.score:.4f} "
            f"type={meta.get('content_type', '?')} "
            f"section={meta.get('section_path', '?')} "
            f"p.{meta.get('page_number', '?')}"
        )
        print(f"    {_preview(nws.node.get_content())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())