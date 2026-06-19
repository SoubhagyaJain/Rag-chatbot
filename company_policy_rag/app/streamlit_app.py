"""
Streamlit entry point for the Company Policy RAG multipage app.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Streamlit puts app/ on sys.path; load path fix before any app.* imports.
_spec = importlib.util.spec_from_file_location(
    "_ensure_path",
    Path(__file__).resolve().parent / "_ensure_path.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod.ensure_project_root()

import streamlit as st

from app.ui.bootstrap import bootstrap_app
from app.ui.theme import inject_global_styles

bootstrap_app()

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

pages = {
    "Employee": [
        st.Page("pages/1_Chat.py", title="Chat", icon="💬", default=True),
    ],
    "Admin": [
        st.Page("pages/2_Documents.py", title="Documents", icon="📄"),
        st.Page("pages/3_System_Health.py", title="System Health", icon="🔧"),
    ],
}

pg = st.navigation(pages)
pg.run()