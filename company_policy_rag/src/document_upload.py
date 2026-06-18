"""Save and index legal PDF uploads from the Streamlit UI."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.indexing import IndexingResult, build_index, configure_llama_index

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_PDF_MAGIC = b"%PDF"


def sanitize_upload_filename(name: str) -> str:
    """Return a safe PDF filename (basename only, allowed chars, .pdf suffix)."""
    base = Path(name).name.strip()
    if not base:
        raise ValueError("Filename is empty")
    stem = Path(base).stem
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")
    if not cleaned:
        cleaned = "document"
    return f"{cleaned}.pdf"


def validate_pdf_content(content: bytes) -> None:
    if not content:
        raise ValueError("Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not content.startswith(_PDF_MAGIC):
        raise ValueError("File is not a valid PDF (missing %PDF header)")


def save_legal_pdf(content: bytes, filename: str) -> Path:
    """Validate and write an uploaded PDF to data/legal/."""
    validate_pdf_content(content)
    safe_name = sanitize_upload_filename(filename)
    settings.ensure_directories()
    dest = settings.legal_dir / safe_name
    dest.write_bytes(content)
    return dest


def list_legal_documents() -> list[dict]:
    """Return metadata for PDFs stored under data/legal/."""
    settings.ensure_directories()
    docs: list[dict] = []
    for path in sorted(settings.legal_dir.glob("*.pdf")):
        stat = path.stat()
        docs.append(
            {
                "filename": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                ),
                "path": str(path),
            }
        )
    return docs


def index_legal_paths(paths: list[Path]) -> IndexingResult:
    """Incrementally index one or more legal PDF paths into Chroma."""
    if not paths:
        raise ValueError("No PDF paths provided for indexing")
    configure_llama_index()
    _, result = build_index(pdf_paths=paths)
    return result