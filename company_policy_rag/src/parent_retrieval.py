"""
Parent-document retrieval: expand ranked child chunks to parent context.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.config import settings
from src.docstore import get_parent_node
from src.timing import get_current_timing, record_stage
from src.utils import timer


def expand_to_parent_documents(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """
    Replace child hits with parent text from docstore (wider context for generation).

    - Deduplicates by parent_id (keeps highest child score)
    - Caption nodes and nodes without parent_id pass through unchanged
    """
    if not settings.enable_parent_document_retrieval or not nodes:
        return nodes

    with timer("parent_expand") as t_expand:
        expanded: list[NodeWithScore] = []
        seen_parents: set[str] = set()

        for nws in nodes:
            meta = nws.node.metadata or {}
            role = meta.get("node_role", "")
            parent_id = meta.get("parent_id")

            if role == "caption" or not parent_id or str(parent_id).startswith("caption_parent_"):
                expanded.append(nws)
                continue

            pid = str(parent_id)
            if pid in seen_parents:
                continue

            parent = get_parent_node(pid)
            if parent is None:
                expanded.append(nws)
                continue

            seen_parents.add(pid)
            child_meta = dict(meta)
            child_meta["expanded_from_child"] = True
            child_meta["retrieved_child_id"] = nws.node.node_id
            child_meta["node_role"] = "parent_expanded"
            for key in ("page_number", "section_path", "section_title", "section_number", "source_file"):
                if meta.get(key) is not None:
                    child_meta[key] = meta[key]

            parent_node = TextNode(
                text=parent.text or "",
                metadata=child_meta,
                id_=parent.node_id,
            )
            expanded.append(NodeWithScore(node=parent_node, score=nws.score))

        result = expanded if expanded else nodes

    if get_current_timing() is not None:
        record_stage("parent_expand", t_expand["elapsed_ms"])

    return result