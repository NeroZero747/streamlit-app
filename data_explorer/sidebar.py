"""Sidebar: brand, data source picker, display controls, footer."""
from __future__ import annotations

import os
from dataclasses import dataclass

import polars as pl
import streamlit as st

from .data_io import list_pins, load_pin, load_uploaded


@dataclass
class SidebarState:
    df: pl.DataFrame | None
    source_label: str
    use_filter_ui: bool
    max_render: int


def _brand_header() -> None:
    st.markdown(
        """
        <div class="sb-header">
          <div class="sb-brand">
            <div class="sb-brand-logo">✦</div>
            <div class="sb-brand-text">
              <span class="sb-brand-name">Data Explorer</span>
              <span class="sb-brand-sub">Polars · Glide · Pins</span>
            </div>
          </div>
        </div>
        <div class="sb-body">
          <div class="sb-group-label">Data source</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _data_source() -> tuple[pl.DataFrame | None, str]:
    source = st.radio(
        "source",
        options=["Upload", "Connect"],
        horizontal=True,
        label_visibility="collapsed",
        key="sb_source",
    )

    df: pl.DataFrame | None = None
    source_label = ""

    if source == "Upload":
        upload = st.file_uploader(
            "file",
            type=["csv", "tsv", "txt", "parquet", "json", "ndjson", "xlsx", "xls"],
            label_visibility="collapsed",
        )
        if upload is not None:
            with st.spinner(f"Parsing {upload.name}…"):
                df = load_uploaded(upload.name, upload.getvalue())
            source_label = upload.name
    else:
        server_url = st.text_input(
            "Server URL",
            value=os.environ.get("CONNECT_SERVER", ""),
            placeholder="https://connect.example.com",
        )
        api_key = st.text_input(
            "API key",
            value=os.environ.get("CONNECT_API_KEY", ""),
            type="password",
        )
        if server_url and api_key:
            try:
                pins_available = list_pins(server_url, api_key)
                pin_name = st.selectbox("Pin", options=[""] + pins_available)
            except Exception as e:
                st.error(f"Connect error: {e}")
                pin_name = st.text_input("Pin name (manual)", value="")
            if pin_name:
                try:
                    df = load_pin(server_url, api_key, pin_name)
                    source_label = f"pin: {pin_name}"
                except Exception as e:
                    st.error(f"Failed to read pin: {e}")
        else:
            st.info("Enter URL + API key to list pins.")

    return df, source_label


def _display_controls(df: pl.DataFrame | None) -> tuple[bool, int]:
    badge = (
        f"<span class='sb-badge'>{df.height:,} rows</span>" if df is not None else ""
    )
    st.markdown(
        f'<div class="sb-group-label" style="margin-top:20px">Display {badge}</div>',
        unsafe_allow_html=True,
    )
    use_filter_ui = st.toggle(
        "Rich filter UI",
        value=True,
        help="Per-column filter widgets above the grid (slower on huge frames).",
    )
    max_render = st.slider(
        "Max rows to render",
        min_value=1_000,
        max_value=200_000,
        value=20_000,
        step=1_000,
        help="The grid is virtualized; only the rendered slice is sent to the browser.",
    )
    return use_filter_ui, max_render


def _footer(df: pl.DataFrame | None) -> None:
    status = "Ready" if df is not None else "Awaiting data"
    st.markdown(
        f"""
        <div class="sb-footer">
          <div class="sb-status"><span class="dot"></span>{status}</div>
          <div class="sb-version">v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> SidebarState:
    with st.sidebar:
        _brand_header()
        df, source_label = _data_source()
        use_filter_ui, max_render = _display_controls(df)
        _footer(df)
    return SidebarState(
        df=df,
        source_label=source_label,
        use_filter_ui=use_filter_ui,
        max_render=max_render,
    )
