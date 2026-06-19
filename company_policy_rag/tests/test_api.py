"""Tests for Aether FastAPI routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agent import AgentTurnResult
from src.api.main import create_app
from src.ollama_client import filter_chat_models, format_model_label


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_root_endpoint(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Aether API"


def test_filter_chat_models_excludes_embed() -> None:
    names = ["qwen2.5:7b", "nomic-embed-text", "llama3.2:latest"]
    filtered = filter_chat_models(names)
    assert "qwen2.5:7b" in filtered
    assert "llama3.2:latest" in filtered
    assert "nomic-embed-text" not in filtered


def test_format_model_label() -> None:
    assert "Qwen2.5" in format_model_label("qwen2.5:7b")


@patch("src.api.routes.health.get_system_health")
def test_health_endpoint(mock_health: MagicMock, client: TestClient) -> None:
    mock_health.return_value = {
        "index_ready": True,
        "chunk_count": 80,
        "collection": "company_policies",
        "last_updated": "2026-06-19",
        "ollama_connected": True,
        "ollama_error": None,
        "llm_model": "qwen2.5:7b",
        "embed_model": "nomic-embed-text",
        "retrieval": {},
    }
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] == 80


@patch("src.api.routes.models.list_available_models")
def test_models_endpoint(mock_models: MagicMock, client: TestClient) -> None:
    mock_models.return_value = {
        "connected": True,
        "error": None,
        "active_model": "qwen2.5:7b",
        "models": [{"id": "qwen2.5:7b", "label": "Qwen2.5 7B"}],
    }
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json()["active_model"] == "qwen2.5:7b"


@patch("src.api.routes.chat.run_chat_turn")
def test_chat_endpoint(mock_chat: MagicMock, client: TestClient) -> None:
    mock_chat.return_value = AgentTurnResult(
        answer="Professional appearance is required.",
        citations=[{"section_path": "Dress and Grooming"}],
        timing={"e2e_ms": 1200},
        low_confidence=False,
        grounding_mode="balanced",
        thinking="Let me check the handbook.",
        retrieval_trace={"chunk_count": 3, "chunks": []},
        message_id="msg-abc",
    )
    resp = client.post(
        "/api/chat",
        json={"message": "What is the dress code?", "corpus_scope": "policy"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Professional appearance" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["thinking"] == "Let me check the handbook."
    assert data["message_id"] == "msg-abc"
    assert data["retrieval_trace"]["chunk_count"] == 3


@patch("src.api.routes.feedback.record_feedback")
def test_feedback_endpoint(mock_record: MagicMock, client: TestClient) -> None:
    mock_record.return_value = {"id": "fb1", "rating": 1}
    resp = client.post(
        "/api/feedback",
        json={
            "rating": 1,
            "question": "Dress code?",
            "answer": "Business casual.",
            "model": "qwen2.5:7b",
            "corpus_scope": "policy",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@patch("src.api.routes.chat.run_chat_stream")
def test_chat_stream_endpoint(mock_stream: MagicMock, client: TestClient) -> None:
    mock_stream.return_value = iter(
        [
            'event: token\ndata: Hello\n\n',
            'event: done\ndata: {"answer":"Hello","citations":[],"low_confidence":false,"grounding_mode":"balanced"}\n\n',
        ]
    )
    resp = client.post(
        "/api/chat/stream",
        json={"message": "Hi", "corpus_scope": "all"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "event: token" in resp.text