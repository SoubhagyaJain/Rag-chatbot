"""Save and index legal PDF uploads from the Streamlit UI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.indexing import (
    IndexingResult,
    build_index,
    configure_llama_index,
    remove_document_from_index,
)
from src.pdf_images import remove_images_for_source

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


@dataclass(frozen=True)
class RemoveDocumentResult:
    filename: str
    file_deleted: bool
    chunks_removed: int


def _resolve_legal_pdf_path(filename: str) -> Path:
    """Return a PDF path under data/legal/; reject path traversal."""
    base = Path(filename).name
    if not base or base != filename.strip() or base != filename:
        raise ValueError("Invalid filename")
    if not base.lower().endswith(".pdf"):
        raise ValueError("Only PDF files can be removed")
    path = (settings.legal_dir / base).resolve()
    legal_root = settings.legal_dir.resolve()
    if path.parent != legal_root:
        raise ValueError("Invalid filename")
    if not path.is_file():
        raise FileNotFoundError(f"Legal PDF not found: {base}")
    return path


def remove_legal_document(filename: str) -> RemoveDocumentResult:
    """Delete a legal PDF from disk and remove its chunks from Chroma."""
    path = _resolve_legal_pdf_path(filename)
    source_file = path.name
    configure_llama_index()
    chunks_removed = remove_document_from_index(source_file)
    remove_images_for_source(source_file)
    path.unlink()
    return RemoveDocumentResult(
        filename=source_file,
        file_deleted=True,
        chunks_removed=chunks_removed,
    )


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