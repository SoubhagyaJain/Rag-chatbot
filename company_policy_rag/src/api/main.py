"""Aether API — FastAPI entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import chat, eval, health, models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_HTML = PROJECT_ROOT / "frontend" / "preview" / "aether.html"
PREVIEW_DIR = PROJECT_ROOT / "frontend" / "preview"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aether API",
        description="Production RAG backend for company policies and AI Agents guidebook",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "null",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api", tags=["chat"])
    app.include_router(models.router, prefix="/api", tags=["models"])
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(eval.router, prefix="/api", tags=["eval"])

    if PREVIEW_DIR.is_dir():
        app.mount("/preview", StaticFiles(directory=str(PREVIEW_DIR)), name="preview")

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "Aether API",
            "docs": "/docs",
            "preview": "/preview/aether.html",
        }

    @app.get("/aether")
    def aether_preview() -> FileResponse:
        if not PREVIEW_HTML.is_file():
            raise HTTPException(status_code=404, detail="frontend/preview/aether.html not found")
        return FileResponse(PREVIEW_HTML)

    return app


app = create_app()