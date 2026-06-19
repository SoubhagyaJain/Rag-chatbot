"""Ollama model listing and switching."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelsResponse, ModelInfo, SetModelRequest
from src.chat_service import apply_llm_model, list_available_models

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    data = list_available_models()
    return ModelsResponse(
        connected=data["connected"],
        error=data.get("error"),
        active_model=data["active_model"],
        models=[ModelInfo.model_validate(m) for m in data["models"]],
    )


@router.put("/models/active")
def set_active_model(request: SetModelRequest) -> dict[str, str]:
    available = list_available_models()
    model_ids = [m["id"] for m in available["models"]]
    if request.model not in model_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not available.",
        )
    apply_llm_model(request.model)
    return {"active_model": request.model}