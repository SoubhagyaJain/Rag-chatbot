"""System health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.chat_service import get_system_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    data = get_system_health()
    return HealthResponse(**data)