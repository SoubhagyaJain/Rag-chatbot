"""One-time app bootstrap (logging, path, Chroma cache)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src.indexing import reset_chroma_client_cache  # noqa: E402
from src.utils import setup_logging  # noqa: E402

_BOOTSTRAPPED = False


def bootstrap_app() -> Path:
    """Initialize logging and clear stale Chroma clients once per process."""
    global _BOOTSTRAPPED
    if not _BOOTSTRAPPED:
        setup_logging("streamlit")
        reset_chroma_client_cache()
        _BOOTSTRAPPED = True
    return PROJECT_ROOT