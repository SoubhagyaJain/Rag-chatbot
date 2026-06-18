"""
Document ingestion pipeline: PDF → text + metadata → chunk → embed → ChromaDB.

Production decisions:
- ChromaDB (persistent) replaces SimpleVectorStore for durable storage, metadata
  filtering (document_type, source_file), and incremental indexing without full rebuilds.
- Page-level documents first, then sentence-aware chunking preserves clause boundaries.
- Hierarchical section detection + propagation keeps citations accurate across chunks.
"""

from __future__ import annotations

import os

# Must be set before chromadb is imported (see src/config.py).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.shared_system_client import SharedSystemClient
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import PROJECT_ROOT, settings
from src.utils import (
    SectionHeading,
    SectionTracker,
    enrich_text_with_section_context,
    infer_category,
    logger,
    parse_section_heading,
    section_metadata_from_context,
    timed,
)


def _headings_from_section_path(section_path: str) -> list[SectionHeading]:
    """
    Reconstruct heading stack from a breadcrumb path for tracker seeding.

    Each segment in section_path (e.g. 'II. GENERAL > A. At-Will') is parsed
    back into SectionHeading objects so node-level propagation stays consistent
    with page-level enrichment.
    """
    headings: list[SectionHeading] = []
    for segment in section_path.split(" > "):
        segment = segment.strip()
        if not segment:
            continue
        parsed = parse_section_heading(segment) or parse_section_heading(
            segment.replace(". ", " ", 1) if ". " in segment else segment
        )
        if parsed:
            headings.append(parsed)
        else:
            headings.append(
                SectionHeading(
                    level=5,
                    section_number=None,
                    section_title=segment,
                    full_label=segment,
                    pattern_name="path_fallback",
                )
            )
    return headings


# ── Result types ───────────────────────────────────────────────────────────


@dataclass
class IndexingResult:
    """Summary returned after an indexing run."""

    documents_loaded: int = 0
    nodes_created: int = 0
    nodes_inserted: int = 0
    nodes_skipped_unchanged: int = 0
    pdf_files_processed: list[str] = field(default_factory=list)
    pdf_files_skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── LlamaIndex global settings ───────────────────────────────────────────────


def configure_llama_index() -> None:
    """Wire Ollama embedding model into LlamaIndex Settings."""
    Settings.embed_model = OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def get_node_parser() -> SentenceSplitter:
    """
    SentenceSplitter respects sentence boundaries — critical for legal text where
    mid-sentence splits destroy clause meaning and hurt retrieval recall.
    """
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        paragraph_separator="\n\n",
    )


# ── ChromaDB vector store ────────────────────────────────────────────────────


def _chroma_settings() -> chromadb.Settings:
    """
    Chroma client settings with telemetry fully disabled.

    anonymized_telemetry=False alone is insufficient: chromadb 0.5.x still calls
    posthog.capture() with a legacy signature that posthog 7.x rejects.
    """
    return chromadb.Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="src.chroma_telemetry.NoOpProductTelemetry",
    )


def reset_chroma_client_cache() -> None:
    """Clear Chroma's in-process client cache (fixes settings conflicts after hot reload)."""
    SharedSystemClient.clear_system_cache()


# Normalize client settings before any PersistentClient is constructed.
_chroma_cfg = _chroma_settings()
chromadb.configure(
    **(_chroma_cfg.model_dump() if hasattr(_chroma_cfg, "model_dump") else _chroma_cfg.dict())
)


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Persistent Chroma client — data survives restarts under storage/chroma/.

    Chroma is preferred over SimpleVectorStore for production RAG because it
    supports metadata filtering (e.g. document_type=legal_document) and stable
    on-disk persistence without custom serialization.
    """
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    path = str(settings.chroma_persist_dir)
    chroma_settings = _chroma_settings()

    try:
        return chromadb.PersistentClient(path=path, settings=chroma_settings)
    except ValueError as exc:
        # Another library may have opened the same persist dir with default settings
        # (e.g. llama_index ChromaVectorStore.from_persist_dir). Clear and retry.
        if "different settings" not in str(exc):
            raise
        logger.warning("Chroma settings conflict for %s — clearing cache and retrying", path)
        reset_chroma_client_cache()
        return chromadb.PersistentClient(path=path, settings=chroma_settings)


def get_chroma_collection(
    client: chromadb.ClientAPI | None = None,
    *,
    recreate: bool = False,
) -> Collection:
    """Get or create the policy documents collection."""
    client = client or get_chroma_client()

    if recreate:
        try:
            client.delete_collection(name=settings.chroma_collection_name)
            logger.info("Deleted Chroma collection: %s", settings.chroma_collection_name)
        except (ValueError, chromadb.errors.NotFoundError, chromadb.errors.InvalidCollectionException):
            pass

    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": settings.chroma_distance_fn},
    )


def get_chroma_vector_store(
    collection: Collection | None = None,
) -> tuple[ChromaVectorStore, StorageContext, Collection]:
    """Build LlamaIndex ChromaVectorStore + StorageContext."""
    collection = collection or get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context, collection


def probe_chroma_index(*, clear_cache: bool = False) -> dict[str, Any]:
    """
    Inspect Chroma health without assuming index_exists() semantics.

    Returns the real chunk count even when the in-process client cache is stale.
    """
    result: dict[str, Any] = {
        "collection": settings.chroma_collection_name,
        "persist_dir": str(settings.chroma_persist_dir),
        "dir_exists": settings.chroma_persist_dir.exists(),
        "sqlite_exists": (settings.chroma_persist_dir / "chroma.sqlite3").exists(),
        "collections": [],
        "count": 0,
        "ready": False,
        "error": None,
    }
    if not result["dir_exists"]:
        result["error"] = "Chroma persist directory does not exist"
        return result

    if clear_cache:
        reset_chroma_client_cache()

    try:
        client = get_chroma_client()
        # Chroma 0.6+ list_collections() returns CollectionName objects; str() is portable.
        result["collections"] = [str(c) for c in client.list_collections()]
        if settings.chroma_collection_name not in result["collections"]:
            result["error"] = (
                f"Collection {settings.chroma_collection_name!r} not found. "
                f"Available: {result['collections'] or 'none'}"
            )
            return result

        collection = client.get_collection(name=settings.chroma_collection_name)
        result["count"] = collection.count()
        result["ready"] = result["count"] > 0
        if not result["ready"]:
            result["error"] = "Collection exists but contains zero chunks"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Chroma index probe failed: %s", exc)

    return result


def index_exists() -> bool:
    """True if the Chroma collection exists and contains at least one chunk."""
    return probe_chroma_index()["ready"]


def get_collection_stats() -> dict[str, Any]:
    """Return Chroma collection name and chunk count (for logging / health checks)."""
    probe = probe_chroma_index()
    return {
        "collection": probe["collection"],
        "count": probe["count"],
        "persist_dir": probe["persist_dir"],
        "ready": probe["ready"],
        "error": probe.get("error"),
        "collections": probe.get("collections", []),
    }


def _sanitize_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Chroma only accepts str, int, float, bool metadata — strip None and coerce.

    Rich policy metadata (section_path, document_type, page_number) must survive
    this step to enable metadata-filtered retrieval later.
    """
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _prepare_nodes_for_chroma(nodes: list[TextNode]) -> list[TextNode]:
    for node in nodes:
        node.metadata = _sanitize_metadata_for_chroma(node.metadata or {})
    return nodes


def _get_indexed_file_hashes(collection: Collection) -> dict[str, str]:
    """Map source_file → file_hash for incremental indexing decisions."""
    if collection.count() == 0:
        return {}

    result = collection.get(include=["metadatas"])
    hashes: dict[str, str] = {}
    for meta in result.get("metadatas") or []:
        if not meta:
            continue
        source_file = meta.get("source_file")
        file_hash = meta.get("file_hash")
        if source_file and file_hash:
            hashes[str(source_file)] = str(file_hash)
    return hashes


def _delete_chunks_for_source(collection: Collection, source_file: str) -> None:
    """Remove all chunks for a source file before re-indexing an updated PDF."""
    try:
        collection.delete(where={"source_file": source_file})
        logger.info("Removed existing chunks for %s", source_file)
    except Exception as exc:
        logger.warning("Could not delete chunks for %s: %s", source_file, exc)


def _filter_paths_for_incremental(
    pdf_paths: list[Path],
    collection: Collection,
    *,
    force_rebuild: bool,
) -> tuple[list[Path], list[Path]]:
    """
    Split paths into (to_index, skipped_unchanged).

    Uses file_hash metadata already stored in Chroma — unchanged files are
    skipped so incremental runs only embed new or modified documents.
    """
    if force_rebuild:
        return pdf_paths, []

    indexed_hashes = _get_indexed_file_hashes(collection)
    to_index: list[Path] = []
    skipped: list[Path] = []

    for path in pdf_paths:
        source_file = path.name
        current_hash = _file_hash(path)
        if indexed_hashes.get(source_file) == current_hash:
            skipped.append(path)
            logger.info("Skipping unchanged file: %s", source_file)
        else:
            to_index.append(path)

    return to_index, skipped


# ── PDF discovery & loading ──────────────────────────────────────────────────


def discover_pdf_files(
    *,
    policies: bool = True,
    legal: bool = True,
) -> list[Path]:
    """Collect PDF paths from configured data directories."""
    pdfs: list[Path] = []
    if policies:
        pdfs.extend(sorted(settings.policies_dir.glob("**/*.pdf")))
    if legal:
        pdfs.extend(sorted(settings.legal_dir.glob("**/*.pdf")))
    return pdfs


def _document_type_for_path(file_path: Path) -> str:
    """Map file location under data/ to a document_type metadata value."""
    try:
        relative = file_path.relative_to(settings.data_dir)
        top_folder = relative.parts[0] if relative.parts else ""
    except ValueError:
        top_folder = ""
    return settings.folder_document_types.get(top_folder, "unknown")


def _parse_page_number(page_label: str | int | None) -> int | None:
    if page_label is None:
        return None
    if isinstance(page_label, int):
        return page_label
    digits = "".join(c for c in str(page_label) if c.isdigit())
    return int(digits) if digits else None


def _file_hash(file_path: Path) -> str:
    """SHA-256 of file bytes — enables change detection for incremental indexing."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sort_key_page(doc: Document) -> tuple[str, int, int]:
    """Sort pages in natural reading order within and across files."""
    page = doc.metadata.get("page_number")
    return (
        doc.metadata.get("source_file", ""),
        page if isinstance(page, int) else 0,
        doc.metadata.get("page_label", 0) if isinstance(doc.metadata.get("page_label"), int) else 0,
    )


def enrich_documents_with_sections(documents: list[Document]) -> list[Document]:
    """Scan pages in reading order and propagate section context onto page metadata."""
    if not settings.enable_section_detection:
        return documents

    by_file: dict[str, list[Document]] = {}
    for doc in documents:
        key = doc.metadata.get("source_file", "")
        by_file.setdefault(key, []).append(doc)

    enriched: list[Document] = []
    for source_file in sorted(by_file):
        tracker = SectionTracker()
        pages = sorted(by_file[source_file], key=_sort_key_page)

        for page_doc in pages:
            context = enrich_text_with_section_context(
                page_doc.text or "",
                tracker,
                scan_max_lines=settings.section_page_scan_lines,
            )
            page_doc.metadata.update(section_metadata_from_context(context))
            enriched.append(page_doc)

        logger.debug(
            "Section context for %s: final path=%s",
            source_file,
            tracker.current_context().section_path,
        )

    return enriched


def load_pdf_documents(file_path: Path) -> list[Document]:
    """Load a single PDF with per-page Documents and base metadata."""
    reader = PDFReader()
    page_docs = reader.load_data(file=file_path)

    document_type = _document_type_for_path(file_path)
    category = infer_category(file_path, document_type)
    source_file = file_path.name
    file_hash = _file_hash(file_path)

    enriched: list[Document] = []
    for page_doc in page_docs:
        page_label = page_doc.metadata.get("page_label") or page_doc.metadata.get("page_number")
        page_number = _parse_page_number(page_label)

        page_doc.metadata.update(
            {
                "source_file": source_file,
                "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                "page_number": page_number,
                "page_label": str(page_label) if page_label else None,
                "document_type": document_type,
                "category": category,
                "file_hash": file_hash,
            }
        )
        enriched.append(page_doc)

    return enrich_documents_with_sections(enriched)


def load_all_documents(
    pdf_paths: list[Path] | None = None,
) -> tuple[list[Document], list[str]]:
    """Load PDFs; return documents and any per-file error messages."""
    paths = pdf_paths or discover_pdf_files()
    all_docs: list[Document] = []
    errors: list[str] = []

    for path in paths:
        try:
            docs = load_pdf_documents(path)
            all_docs.extend(docs)
            logger.info("Loaded %d pages from %s", len(docs), path.name)
        except Exception as exc:
            msg = f"Failed to load {path}: {exc}"
            logger.error(msg)
            errors.append(msg)

    return all_docs, errors


# ── Chunking & node-level section propagation ────────────────────────────────


def _sort_key_node(node: TextNode) -> tuple[str, int, str]:
    """Order chunks in reading order for section propagation."""
    meta = node.metadata or {}
    page = meta.get("page_number")
    return (
        meta.get("source_file", ""),
        page if isinstance(page, int) else 0,
        node.node_id or "",
    )


def enrich_nodes_with_sections(nodes: list[TextNode]) -> list[TextNode]:
    """Propagate hierarchical section metadata to every chunk."""
    if not settings.enable_section_detection:
        return nodes

    by_file: dict[str, list[TextNode]] = {}
    for node in nodes:
        key = (node.metadata or {}).get("source_file", "")
        by_file.setdefault(key, []).append(node)

    for source_file in sorted(by_file):
        tracker = SectionTracker()
        file_nodes = sorted(by_file[source_file], key=_sort_key_node)

        last_page: int | None = None
        for node in file_nodes:
            page_meta = node.metadata or {}
            page_num = page_meta.get("page_number")
            if isinstance(page_num, int) and page_num != last_page:
                last_page = page_num
                if not tracker.current_context().section_path and page_meta.get("section_path"):
                    for heading in _headings_from_section_path(page_meta["section_path"]):
                        tracker.update(heading)

            context = enrich_text_with_section_context(
                node.text or "",
                tracker,
                scan_max_lines=None,
            )

            section_meta = section_metadata_from_context(context)
            if not section_meta.get("section_title"):
                for key in ("section_title", "section_number", "section_path", "section_level"):
                    if page_meta.get(key) is not None:
                        section_meta[key] = page_meta[key]

            node.metadata.update(section_meta)

    return nodes


def documents_to_nodes(documents: list[Document]) -> list[TextNode]:
    """Parse Documents into Chroma-ready TextNodes with section metadata."""
    parser = get_node_parser()
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    nodes = enrich_nodes_with_sections(nodes)
    return _prepare_nodes_for_chroma(nodes)


# ── Index build / load ───────────────────────────────────────────────────────


def load_index() -> VectorStoreIndex:
    """
    Load VectorStoreIndex from the persisted Chroma collection.

    Uses from_vector_store — no separate SimpleVectorStore persist dir needed.
    """
    configure_llama_index()

    if not index_exists():
        raise FileNotFoundError(
            f"No index found in Chroma collection '{settings.chroma_collection_name}' "
            f"at {settings.chroma_persist_dir}. Run: python scripts/index_documents.py"
        )

    vector_store, storage_context, _ = get_chroma_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    logger.info(
        "Loaded Chroma index | collection=%s | chunks=%d",
        settings.chroma_collection_name,
        get_collection_stats()["count"],
    )
    return index


def get_or_create_index(*, rebuild_if_missing: bool = True) -> VectorStoreIndex:
    """Load existing Chroma index or build from documents if missing."""
    if index_exists():
        return load_index()
    if rebuild_if_missing:
        index, _ = build_index()
        return index
    raise FileNotFoundError("Index not found and rebuild_if_missing=False")


@timed("build_index")
def build_index(
    pdf_paths: list[Path] | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[VectorStoreIndex, IndexingResult]:
    """
    Indexing pipeline with Chroma persistence and incremental support.

    - force_rebuild: wipe collection and re-index all provided PDFs.
    - default (incremental): skip unchanged files (by file_hash), append/update others.
    """
    configure_llama_index()
    result = IndexingResult()

    if force_rebuild and settings.chroma_persist_dir.exists():
        shutil.rmtree(settings.chroma_persist_dir)
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared Chroma persist dir: %s", settings.chroma_persist_dir)

    all_paths = pdf_paths or discover_pdf_files()
    if not all_paths:
        logger.warning("No PDF paths provided for indexing.")

    client = get_chroma_client()
    collection = get_chroma_collection(client, recreate=force_rebuild)
    vector_store, storage_context, collection = get_chroma_vector_store(collection)

    paths_to_index, paths_skipped = _filter_paths_for_incremental(
        all_paths, collection, force_rebuild=force_rebuild
    )
    result.pdf_files_skipped = [p.name for p in paths_skipped]
    result.nodes_skipped_unchanged = len(paths_skipped)

    if not paths_to_index:
        if index_exists():
            logger.info("All files unchanged — using existing Chroma index.")
            return load_index(), result
        logger.warning("No documents to index. Place PDFs in data/policies/ or data/legal/.")
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context), result

    documents, errors = load_all_documents(paths_to_index)
    result.errors = errors
    result.documents_loaded = len(documents)
    result.pdf_files_processed = list(
        {d.metadata.get("source_file", "") for d in documents if d.metadata.get("source_file")}
    )

    if not documents:
        if index_exists():
            return load_index(), result
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context), result

    # Remove stale chunks for files being updated (hash changed or new file)
    for path in paths_to_index:
        _delete_chunks_for_source(collection, path.name)

    nodes = documents_to_nodes(documents)
    result.nodes_created = len(nodes)
    logger.info("Created %d chunks from %d page-documents", len(nodes), len(documents))

    if index_exists() and not force_rebuild:
        index = load_index()
        index.insert_nodes(nodes)
        result.nodes_inserted = len(nodes)
        logger.info("Incrementally inserted %d nodes into Chroma", len(nodes))
    else:
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        result.nodes_inserted = len(nodes)
        logger.info(
            "Created Chroma index | collection=%s | dir=%s",
            settings.chroma_collection_name,
            settings.chroma_persist_dir,
        )

    stats = get_collection_stats()
    logger.info("Chroma collection '%s' now has %d chunks", stats["collection"], stats["count"])

    return index, result


# ── Query engine factory (delegates to retriever.py for reranking) ─────────────


def create_query_engine(
    index: VectorStoreIndex | None = None,
    *,
    filters: dict[str, Any] | None = None,
):
    """
    Create a retriever-backed query engine over the Chroma index.

    When ENABLE_RERANKER=true, over-retrieves then cross-encoder reranks.
    filters: optional Chroma metadata filter, e.g. {"document_type": "legal_document"}
    """
    from src.retriever import build_query_engine

    return build_query_engine(index, filters=filters)


def create_retriever(
    index: VectorStoreIndex | None = None,
    *,
    filters: dict[str, Any] | None = None,
):
    """
    Expose Chroma retriever for citation extraction and metadata-filtered search.

    Applies reranker post-processor when enabled (see src/retriever.py).
    Example filter for legal docs only:
        filters={"document_type": "legal_document"}
    """
    from src.retriever import build_retriever

    return build_retriever(index, filters=filters)