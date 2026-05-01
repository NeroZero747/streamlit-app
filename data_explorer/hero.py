"""Hero header — empty state and loaded state."""
from __future__ import annotations

import html

import polars as pl
import streamlit as st


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-row">
            <div>
              <div class="hero-eyebrow">Workspace</div>
              <div class="hero-title">Data Explorer</div>
              <div class="hero-sub">Upload a file or connect to a Posit Connect pin to start exploring.</div>
              <div style="margin-top:14px">
                <span class='pill'>Polars</span>
                <span class='pill cyan'>Glide Data Grid</span>
                <span class='pill pink'>Plotly</span>
                <span class='pill cyan'>Posit Pins</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='glass' style='margin-top:24px;text-align:center;padding:60px 24px'>
          <div style="font-size:48px;margin-bottom:8px">📂</div>
          <h3 style="margin:0">No data loaded yet</h3>
          <p style='color:#94a3b8;margin-top:6px'>
            Use the sidebar to upload CSV, TSV, Parquet, JSON/NDJSON, or Excel — up to 5 GB.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(df: pl.DataFrame, source_label: str) -> None:
    label = html.escape(source_label) if source_label else "—"
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-row">
            <div>
              <div class="hero-eyebrow">Workspace</div>
              <div class="hero-title">Data Explorer</div>
              <div class="hero-sub">Inspect, filter, visualize, and export your dataset.</div>
              <div class="hero-source"><span class="dot"></span>{label}</div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;align-self:flex-start">
              <span class='pill'>Polars</span>
              <span class='pill cyan'>Glide</span>
              <span class='pill pink'>Plotly</span>
            </div>
          </div>
          <div class="metrics">
            <div class="metric">
              <div class="metric-label">Rows</div>
              <div class="metric-value">{df.height:,}</div>
            </div>
            <div class="metric">
              <div class="metric-label">Columns</div>
              <div class="metric-value">{df.width}</div>
            </div>
            <div class="metric">
              <div class="metric-label">Memory</div>
              <div class="metric-value">{df.estimated_size("mb"):.1f}<span class="unit">MB</span></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
