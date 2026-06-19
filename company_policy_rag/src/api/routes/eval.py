"""Evaluation job endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import EvalJobResponse, EvalRunRequest
from src.chat_service import get_eval_job, start_eval_job

router = APIRouter()


@router.post("/eval/run", response_model=EvalJobResponse)
def run_eval(request: EvalRunRequest) -> EvalJobResponse:
    job = start_eval_job(max_samples=request.max_samples)
    return EvalJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        result=job.result,
        error=job.error,
    )


@router.get("/eval/{job_id}", response_model=EvalJobResponse)
def eval_status(job_id: str) -> EvalJobResponse:
    job = get_eval_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Eval job not found")
    return EvalJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        result=job.result,
        error=job.error,
    )