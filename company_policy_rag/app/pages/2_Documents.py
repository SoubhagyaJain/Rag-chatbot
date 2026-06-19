"""Admin document management page."""

from __future__ import annotations

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