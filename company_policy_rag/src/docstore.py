"""
Parent-node docstore for hierarchical chunking.

Parents are not embedded in Chroma; children carry parent_id for Phase 2
parent-document retrieval. Persisted as JSON under storage/docstore/.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode

from src.config import settings
from src.utils import logger

PARENT_STORE_FILE = "parent_nodes.json"


def _store_path() -> Path:
    return settings.docstore_dir / PARENT_STORE_FILE


def _empty_store() -> dict[str, Any]:
    return {"parents": {}, "by_source_file": {}}


def load_parent_store() -> dict[str, Any]:
    path = _store_path()
    if not path.is_file():
        return _empty_store()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt parent docstore at %s — resetting", path)
        return _empty_store()


def save_parent_store(store: dict[str, Any]) -> None:
    settings.docstore_dir.mkdir(parents=True, exist_ok=True)
    _store_path().write_text(json.dumps(store, indent=2), encoding="utf-8")


def new_parent_id() -> str:
    return f"parent_{uuid.uuid4().hex[:12]}"


def register_parent_nodes(parent_nodes: list[TextNode]) -> None:
    """Upsert parent nodes into the docstore."""
    if not parent_nodes:
        return

    store = load_parent_store()
    parents = store.setdefault("parents", {})
    by_source = store.setdefault("by_source_file", {})

    for node in parent_nodes:
        meta = node.metadata or {}
        pid = node.node_id or new_parent_id()
        node.node_id = pid
        parents[pid] = {
            "text": node.text or "",
            "metadata": {k: v for k, v in meta.items() if v is not None},
            "child_ids": meta.get("child_ids", []),
        }
        source_file = str(meta.get("source_file", ""))
        if source_file:
            ids: list[str] = by_source.setdefault(source_file, [])
            if pid not in ids:
                ids.append(pid)

    save_parent_store(store)
    logger.info("Registered %d parent nodes in docstore", len(parent_nodes))


def remove_parents_for_source(source_file: str) -> int:
    """Delete all parent nodes for a source file."""
    store = load_parent_store()
    parents = store.get("parents", {})
    by_source = store.get("by_source_file", {})
    pids = by_source.pop(source_file, [])
    removed = 0
    for pid in pids:
        if pid in parents:
            del parents[pid]
            removed += 1
    save_parent_store(store)
    if removed:
        logger.info("Removed %d parent nodes for %s", removed, source_file)
    return removed


def get_parent_node(parent_id: str) -> TextNode | None:
    """Load a parent node by id (for Phase 2 parent-document retrieval)."""
    store = load_parent_store()
    entry = store.get("parents", {}).get(parent_id)
    if not entry:
        return None
    meta = dict(entry.get("metadata") or {})
    meta["node_role"] = "parent"
    meta["child_ids"] = entry.get("child_ids", [])
    return TextNode(text=entry.get("text", ""), metadata=meta, id_=parent_id)


def clear_parent_store() -> None:
    """Wipe all parent nodes (used on force rebuild)."""
    path = _store_path()
    if path.exists():
        path.unlink()
    logger.info("Cleared parent docstore")