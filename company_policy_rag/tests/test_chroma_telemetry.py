"""Chroma telemetry must not emit posthog API errors."""

from __future__ import annotations

import logging

from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.telemetry.product import ProductTelemetryClient

from src.chroma_telemetry import NoOpProductTelemetry
from src.indexing import (
    _chroma_settings,
    get_chroma_client,
    get_chroma_collection,
    index_exists,
    probe_chroma_index,
)


def test_chroma_settings_use_noop_telemetry() -> None:
    chroma_settings = _chroma_settings()
    assert chroma_settings.anonymized_telemetry is False
    assert chroma_settings.chroma_product_telemetry_impl.endswith("NoOpProductTelemetry")


def test_chroma_client_emits_no_telemetry_errors(caplog) -> None:
    SharedSystemClient.clear_system_cache()
    caplog.set_level(logging.ERROR)

    client = get_chroma_client()
    collection = get_chroma_collection(client)

    telemetry_errors = [
        record.message
        for record in caplog.records
        if "Failed to send telemetry event" in record.message
    ]
    assert telemetry_errors == []
    assert collection.name


def test_noop_telemetry_is_registered() -> None:
    SharedSystemClient.clear_system_cache()
    client = get_chroma_client()
    telemetry = client._system.instance(ProductTelemetryClient)  # type: ignore[attr-defined]
    assert isinstance(telemetry, NoOpProductTelemetry)


def test_index_exists_recovers_from_settings_conflict() -> None:
    from src.config import settings

    SharedSystemClient.clear_system_cache()
    path = str(settings.chroma_persist_dir)

    import chromadb

    # Simulate another library opening the persist dir with default settings.
    chromadb.PersistentClient(path=path, settings=chromadb.Settings())

    # Our client should recover from the settings conflict.
    client = get_chroma_client()
    collection = get_chroma_collection(client)
    if collection.count() == 0:
        collection.add(
            ids=["telemetry-conflict-probe"],
            documents=["settings conflict recovery probe"],
            metadatas=[{"source": "test_chroma_telemetry"}],
        )

    assert index_exists() is True


def test_probe_reports_actual_chunk_count() -> None:
    probe = probe_chroma_index(clear_cache=True)
    assert probe["dir_exists"] is True
    assert probe["collection"] == "company_policies"
    if probe["ready"]:
        assert probe["count"] > 0
        assert probe["error"] is None