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

import streamlit as st

from data_explorer.hero import render_empty_state, render_hero
from data_explorer.sidebar import render_sidebar
from data_explorer.tabs import charts_fragment, grid_fragment, schema_fragment
from data_explorer.theme import configure_page, inject_theme


def main() -> None:
    configure_page()
    inject_theme()

    # ---- Company logo — top-right navbar ----------------------------------------
    # Replace "Acme<em>Corp</em>" with your brand name; swap the SVG mark as needed.
    st.markdown(
        """
        <div class="co-topbar-logo">
          <svg class="co-mark" viewBox="0 0 20 20" fill="none"
               xmlns="http://www.w3.org/2000/svg">
            <path d="M10 2L17.3 6.25V13.75L10 18L2.7 13.75V6.25L10 2Z"
                  fill="url(#co_g)"/>
            <path d="M7 10.5L9.2 12.7L13.3 8.3"
                  stroke="white" stroke-width="1.6"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <defs>
              <linearGradient id="co_g" x1="2.7" y1="2" x2="17.3" y2="18"
                              gradientUnits="userSpaceOnUse">
                <stop stop-color="#cb187d"/>
                <stop offset="1" stop-color="#a855f7"/>
              </linearGradient>
            </defs>
          </svg>
          <span class="co-wordmark">Acme<em>Corp</em></span>
        </div>
        """,
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
