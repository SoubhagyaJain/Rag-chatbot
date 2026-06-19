"""Chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.chat_service import run_chat_turn, turn_to_dict

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        turn = run_chat_turn(
            request.message.strip(),
            corpus_scope=request.corpus_scope,
            chat_mode=request.chat_mode,
            llm_model=request.llm_model,
            grounding_mode=request.grounding_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data = turn_to_dict(turn)
    return ChatResponse(**data)