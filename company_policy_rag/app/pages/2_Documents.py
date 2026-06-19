"""Admin document management page."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_ensure_path",
    Path(__file__).resolve().parents[1] / "_ensure_path.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod.ensure_project_root()

import streamlit as st

from app.ui.bootstrap import bootstrap_app
from app.ui.components.documents import render_documents_page
from app.ui.sidebar import render_sidebar_controls, render_sidebar_status
from app.ui.theme import inject_global_styles

bootstrap_app()
inject_global_styles()
render_sidebar_controls()
render_sidebar_status()
render_documents_page()