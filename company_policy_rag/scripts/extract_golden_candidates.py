#!/usr/bin/env python3
"""Scan indexed content for golden-set candidate keywords (code, diagrams, sections)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.indexing import configure_llama_index, index_exists, load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract golden-set keyword candidates from Chroma")
    parser.add_argument("--top", type=int, default=30, help="Max lines to print per category")
    args = parser.parse_args()

    if not index_exists():
        print("No index found. Run: python scripts/index_documents.py --force")
        raise SystemExit(1)

    configure_llama_index()
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=200)
    nodes = retriever.retrieve("code tool diagram building blocks workflow pattern")

    code_hits: list[str] = []
    diagram_hits: list[str] = []
    section_hits: list[str] = []

    for nws in nodes:
        text = nws.get_content() or ""
        meta = nws.metadata or {}
        label = f"{meta.get('source_file')} p.{meta.get('page_number')} [{meta.get('content_type')}]"
        snippet = text[:120].replace("\n", " ")
        if meta.get("content_type") == "code" or "[CODE BLOCK" in text:
            code_hits.append(f"{label}: {snippet}")
        if meta.get("content_type") == "diagram_caption" or re.search(r"figure|diagram", text, re.I):
            diagram_hits.append(f"{label}: {snippet}")
        if meta.get("section_path"):
            section_hits.append(f"{label}: {meta.get('section_path')}")

    print("=== Code chunks ===")
    for line in code_hits[: args.top]:
        print(line)
    print("\n=== Diagram / caption chunks ===")
    for line in diagram_hits[: args.top]:
        print(line)
    print("\n=== Section paths (sample) ===")
    for line in section_hits[: args.top]:
        print(line)
    print(f"\nIndex: {settings.chroma_collection_name} @ {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()