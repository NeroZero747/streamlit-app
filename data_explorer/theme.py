"""Page configuration and CSS theme loading."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "styles.css"


def configure_page() -> None:
    st.set_page_config(
        page_title="Data Explorer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_theme() -> None:
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
