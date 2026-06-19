"""Add project root to sys.path (importable without the app.* prefix)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Insert company_policy_rag root so `app.*` and `src.*` imports resolve."""
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root