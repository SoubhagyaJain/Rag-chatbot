"""Console entry points for the soubhagya-policy-rag PyPI package."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def _script_candidates(script: str) -> list[Path]:
    package_root = Path(__file__).resolve().parent.parent
    return [
        package_root / "scripts" / f"{script}.py",
        Path.cwd() / "scripts" / f"{script}.py",
    ]


def _load_script_main(script: str) -> Callable[..., int]:
    for path in _script_candidates(script):
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(f"_policy_rag_{script}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        main_fn = getattr(module, "main", None)
        if callable(main_fn):
            return main_fn
    raise SystemExit(
        f"Could not find scripts/{script}.py. "
        "Run from the project directory or set POLICY_RAG_ROOT."
    )


def _streamlit_app_path() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    for candidate in (
        package_root / "app" / "streamlit_app.py",
        Path.cwd() / "app" / "streamlit_app.py",
    ):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "streamlit_app.py not found. Clone the repo or set POLICY_RAG_ROOT to the project root."
    )


def index_main() -> None:
    """Index PDFs from data/policies/ and data/legal/."""
    raise SystemExit(_load_script_main("index_documents")())


def eval_main() -> None:
    """Run golden-set evaluation."""
    raise SystemExit(_load_script_main("evaluate")())


def chat_main() -> None:
    """Launch the Streamlit chat UI."""
    app_path = _streamlit_app_path()
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
            ]
        )
    )