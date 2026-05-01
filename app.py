"""⚡ Data Explorer — Streamlit + Polars + Glide Data Grid + Plotly.

Reactivity / speed strategy:
  * st.dataframe is backed by Glide Data Grid (canvas-rendered, virtualized) —
    handles huge frames smoothly with zero plug-in overhead.
  * Filtering happens in Polars (lazy + fast); only the resulting view (head N)
    is sent to the browser. No 100 MB+ payloads.
  * @st.cache_data memoizes IO and the filter pipeline.
  * @st.fragment isolates each tab so widget changes don't rerun the whole app.

Modular layout:
  app.py                       — entry point (this file)
  assets/styles.css            — full app theme / CSS
  data_explorer/
    theme.py                   — page config + CSS injection
    data_io.py                 — cached loaders (uploads, Posit Pins)
    schema.py                  — schema_summary, head_pandas, column_filter_meta
    filters.py                 — polars_filter_ui (per-column filter UI)
    sidebar.py                 — render_sidebar() → SidebarState
    hero.py                    — render_empty_state, render_hero
    downloads.py               — save_as_button + icon SVGs
    tabs/
      grid.py                  — grid_fragment
      charts.py                — charts_fragment
      schema_tab.py            — schema_fragment
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from data_explorer.hero import render_empty_state, render_hero
from data_explorer.sidebar import render_sidebar
from data_explorer.tabs import charts_fragment, grid_fragment, schema_fragment
from data_explorer.theme import configure_page, inject_theme


def _logo_html() -> str:
    """Return an <img> tag with the company logo embedded as base64, or '' if not found."""
    assets = Path(__file__).parent / "assets"
    for name in ("logo.png", "logo.jpg", "logo.jpeg", "logo.svg"):
        path = assets / name
        if path.exists():
            mime = "image/svg+xml" if name.endswith(".svg") else f"image/{path.suffix.lstrip('.')}"
            data = base64.b64encode(path.read_bytes()).decode()
            return f'<img class="co-logo-img" src="data:{mime};base64,{data}" alt="logo" />'
    return ""


def main() -> None:
    configure_page()
    inject_theme()

    # ---- Company logo — top-right navbar ----------------------------------------
    logo = _logo_html()
    if logo:
        st.markdown(
            f'<div class="co-topbar-logo">{logo}</div>',
            unsafe_allow_html=True,
        )

    state = render_sidebar()

    if state.df is None:
        render_empty_state()
        st.stop()

    # Stash for fragment scopes
    st.session_state["_df"] = state.df
    st.session_state["_source_label"] = state.source_label
    st.session_state["_max_render"] = state.max_render

    render_hero(state.df, state.source_label)

    tab_grid, tab_charts, tab_schema = st.tabs(
        ["  Grid  ", "  Pivot Table  ", "  Schema  "]
    )
    with tab_grid:
        grid_fragment()
    with tab_charts:
        charts_fragment()
    with tab_schema:
        schema_fragment()


main()
