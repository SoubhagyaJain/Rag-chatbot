#!/usr/bin/env python3
"""
CLI script to index PDF documents from data/policies/ and data/legal/.

Usage:
    python scripts/index_documents.py
    python scripts/index_documents.py --force
    python scripts/index_documents.py --policies-only
    python scripts/index_documents.py --file data/policies/employee_handbook.pdf

Also importable:
    from scripts.index_documents import main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.indexing import build_index, discover_pdf_files, get_collection_stats
from src.utils import logger, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index company policy and legal PDFs into the vector store.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index and rebuild from scratch.",
    )
    parser.add_argument(
        "--policies-only",
        action="store_true",
        help="Index only data/policies/ (skip legal/).",
    )
    parser.add_argument(
        "--legal-only",
        action="store_true",
        help="Index only data/legal/ (skip policies/).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        dest="files",
        help="Index specific PDF file(s) instead of scanning directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List PDFs that would be indexed without building the index.",
    )
    return parser.parse_args(argv)


def resolve_pdf_paths(args: argparse.Namespace) -> list[Path]:
    if args.files:
        paths = []
        for f in args.files:
            p = f if f.is_absolute() else PROJECT_ROOT / f
            if not p.exists():
                logger.error("File not found: %s", p)
                continue
            paths.append(p)
        return paths

    include_policies = not args.legal_only
    include_legal = not args.policies_only
    return discover_pdf_files(policies=include_policies, legal=include_legal)


def main(argv: list[str] | None = None) -> int:
    setup_logging("index_documents")
    args = parse_args(argv)

    pdf_paths = resolve_pdf_paths(args)

    if not pdf_paths:
        logger.error(
            "No PDF files found. Add files to:\n"
            "  - %s\n"
            "  - %s",
            settings.policies_dir,
            settings.legal_dir,
        )
        return 1

    logger.info("Found %d PDF(s) to index", len(pdf_paths))
    for p in pdf_paths:
        logger.info("  • %s", p.relative_to(PROJECT_ROOT))

    if args.dry_run:
        logger.info("Dry run — exiting without indexing.")
        return 0

    index, result = build_index(pdf_paths, force_rebuild=args.force)

    logger.info("─" * 50)
    logger.info("Indexing complete")
    logger.info("  Documents (pages): %d", result.documents_loaded)
    logger.info("  Chunks created:    %d", result.nodes_created)
    logger.info("  PDF files:         %s", ", ".join(result.pdf_files_processed) or "none")
    stats = get_collection_stats()
    logger.info("  Chroma collection: %s", stats["collection"])
    logger.info("  Chroma persist:    %s", stats["persist_dir"])
    logger.info("  Total chunks:      %d", stats["count"])
    if result.pdf_files_skipped:
        logger.info("  Skipped (unchanged): %s", ", ".join(result.pdf_files_skipped))

    if result.errors:
        logger.warning("Errors (%d):", len(result.errors))
        for err in result.errors:
            logger.warning("  %s", err)
        return 1

    # Touch index to confirm it loaded
    _ = index
    return 0


if __name__ == "__main__":
    raise SystemExit(main())