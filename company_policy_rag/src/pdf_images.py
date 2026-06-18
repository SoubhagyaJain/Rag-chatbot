"""Extract and serve PDF page images for citation display."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.config import settings
from src.utils import logger

BY_SOURCE_FILE = "by_source.json"
MANIFEST_FILE = "manifest.json"
MAX_IMAGES_PER_CITATION = 4


def images_dir_for_hash(file_hash: str) -> Path:
    return settings.pdf_images_dir / file_hash


def _by_source_path() -> Path:
    return settings.pdf_images_dir / BY_SOURCE_FILE


def _load_by_source() -> dict[str, str]:
    path = _by_source_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt PDF image registry at %s — resetting", path)
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _save_by_source(mapping: dict[str, str]) -> None:
    settings.pdf_images_dir.mkdir(parents=True, exist_ok=True)
    _by_source_path().write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def _register_source(source_file: str, file_hash: str) -> None:
    mapping = _load_by_source()
    mapping[source_file] = file_hash
    _save_by_source(mapping)


def _pixmap_to_rgb(pix: Any) -> Any:
    import fitz

    if pix.n - pix.alpha < 4:
        return pix
    return fitz.Pixmap(fitz.csRGB, pix)


def _save_pixmap(pix: Any, dest: Path) -> None:
    rgb = _pixmap_to_rgb(pix)
    try:
        rgb.save(str(dest))
    finally:
        if rgb is not pix:
            rgb = None


def extract_pdf_images(
    pdf_path: Path,
    *,
    file_hash: str,
    source_file: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Extract embedded images per page and a page thumbnail when embeds exist.

    Writes manifest.json under storage/images/{file_hash}/.
    """
    if not settings.enable_pdf_images:
        return {}

    import fitz

    source_name = source_file or pdf_path.name
    settings.pdf_images_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = images_dir_for_hash(file_hash)
    manifest_path = dest_dir / MANIFEST_FILE

    if not force and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("file_hash") == file_hash and existing.get("source_file") == source_name:
                _register_source(source_name, file_hash)
                logger.info("PDF images unchanged for %s — skipping extraction", source_name)
                return existing
        except json.JSONDecodeError:
            pass

    if force and dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    min_px = settings.pdf_image_min_px
    thumb_dpi = settings.pdf_page_thumb_dpi
    pages: dict[str, dict[str, Any]] = {}
    total_embedded = 0

    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            embedded_names: list[str] = []

            for img_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                try:
                    base_pix = fitz.Pixmap(doc, xref)
                except Exception as exc:
                    logger.warning(
                        "Could not read image xref=%s on page %d of %s: %s",
                        xref,
                        page_number,
                        source_name,
                        exc,
                    )
                    continue

                if base_pix.width < min_px or base_pix.height < min_px:
                    base_pix = None
                    continue

                filename = f"page_{page_number}_img_{img_index}.png"
                out_path = dest_dir / filename
                try:
                    _save_pixmap(base_pix, out_path)
                    embedded_names.append(filename)
                    total_embedded += 1
                except Exception as exc:
                    logger.warning(
                        "Could not save image on page %d of %s: %s",
                        page_number,
                        source_name,
                        exc,
                    )
                finally:
                    base_pix = None

            thumbnail_name: str | None = None
            if embedded_names:
                thumb_path = dest_dir / f"page_{page_number}_thumb.png"
                thumb_pix = None
                try:
                    thumb_pix = page.get_pixmap(dpi=thumb_dpi)
                    _save_pixmap(thumb_pix, thumb_path)
                    thumbnail_name = thumb_path.name
                except Exception as exc:
                    logger.warning(
                        "Could not render thumbnail for page %d of %s: %s",
                        page_number,
                        source_name,
                        exc,
                    )

            if embedded_names or thumbnail_name:
                pages[str(page_number)] = {
                    "embedded": embedded_names,
                    "thumbnail": thumbnail_name,
                }
    finally:
        doc.close()

    manifest = {
        "source_file": source_name,
        "file_hash": file_hash,
        "pages": pages,
        "embedded_count": total_embedded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _register_source(source_name, file_hash)
    logger.info(
        "Extracted %d embedded image(s) across %d page(s) from %s",
        total_embedded,
        len(pages),
        source_name,
    )
    return manifest


def get_page_images(source_file: str, page_number: int | None) -> list[Path]:
    """Return image paths for a citation (embedded first, then page thumbnail)."""
    if not settings.enable_pdf_images or page_number is None:
        return []

    file_hash = _load_by_source().get(source_file)
    if not file_hash:
        return []

    manifest_path = images_dir_for_hash(file_hash) / MANIFEST_FILE
    if not manifest_path.is_file():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    page_data = manifest.get("pages", {}).get(str(page_number))
    if not page_data:
        return []

    base = images_dir_for_hash(file_hash)
    paths: list[Path] = []
    for name in page_data.get("embedded") or []:
        candidate = base / name
        if candidate.is_file():
            paths.append(candidate)

    thumbnail = page_data.get("thumbnail")
    if thumbnail:
        candidate = base / thumbnail
        if candidate.is_file() and candidate not in paths:
            paths.append(candidate)

    return paths


def remove_images_for_source(source_file: str) -> None:
    """Delete cached PDF images when a source file is removed."""
    mapping = _load_by_source()
    file_hash = mapping.pop(source_file, None)
    if not file_hash:
        return

    _save_by_source(mapping)
    dest_dir = images_dir_for_hash(file_hash)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
        logger.info("Removed PDF image cache for %s", source_file)


def clear_all_pdf_images() -> None:
    """Remove all extracted PDF images (e.g. on full index rebuild)."""
    if settings.pdf_images_dir.exists():
        shutil.rmtree(settings.pdf_images_dir)
    settings.pdf_images_dir.mkdir(parents=True, exist_ok=True)