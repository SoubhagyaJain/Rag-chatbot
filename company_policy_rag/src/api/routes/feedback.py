"""Answer feedback endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import FeedbackRequest
from src.chat_service import feedback_summary
from src.feedback_store import record_feedback

router = APIRouter()


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest) -> dict[str, object]:
    entry = record_feedback(
        rating=request.rating,
        question=request.question,
        answer=request.answer,
        model=request.model,
        corpus_scope=request.corpus_scope,
        message_id=request.message_id,
        comment=request.comment,
    )
    return {"ok": True, "id": entry["id"]}


@router.get("/feedback/summary")
def get_feedback_summary() -> dict[str, object]:
    return feedback_summary()