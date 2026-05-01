"""Sidebar: brand, data source picker, display controls, footer."""
from __future__ import annotations

import functools
import os
import urllib.request
from dataclasses import dataclass

import polars as pl
import streamlit as st

from .data_io import get_pin_files, list_pins, load_pin, load_pin_file, load_uploaded


@dataclass
class SidebarState:
    df: pl.DataFrame | None
    source_label: str
    max_render: int


# ---- Iconify icon loader ---------------------------------------------------
# Icons are fetched from the Iconify CDN on first use and cached for the
# lifetime of the Python process (across Streamlit reruns).
# Browse icons at https://icon-sets.iconify.design/

@functools.lru_cache(maxsize=32)
def _icon(name: str) -> str:
    """Return an inline SVG for `name` (e.g. 'lucide:database') via Iconify API."""
    prefix, icon_name = name.split(":", 1)
    url = (
        f"https://api.iconify.design/{prefix}/{icon_name}.svg"
        "?color=currentColor"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "streamlit-data-explorer/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        # Minimal fallback square so layout never breaks
        return '<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" xmlns="http://www.w3.org/2000/svg"></svg>'


def _brand() -> None:
    icon = _icon("lucide:bar-chart-2")
    st.markdown(
        f'<div class="nav-brand">'
        f'<span class="nav-brand-icon">{icon}</span>'
        f'<span class="nav-brand-title">Data Explorer</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _section_label(text: str) -> None:
    """Consistent section label above a group of controls."""
    st.markdown(f'<p class="nav-label">{text}</p>', unsafe_allow_html=True)


def _data_source() -> tuple[pl.DataFrame | None, str]:
    _section_label("Data source")

    source = st.radio(
        "source",
        options=["File Upload", "Posit Connect"],
        horizontal=True,
        label_visibility="collapsed",
        key="sb_source",
    )

    df: pl.DataFrame | None = None
    source_label = ""

    if source == "File Upload":
        upload = st.file_uploader(
            "file",
            type=["csv", "tsv", "txt", "parquet", "json", "ndjson", "xlsx", "xls"],
            label_visibility="collapsed",
        )
        if upload is not None:
            with st.spinner(f"Parsing {upload.name}\u2026"):
                df = load_uploaded(upload.name, upload.getvalue())
            source_label = upload.name
            file_icon = _icon("lucide:file-text")
            st.markdown(
                f"""
                <div class="nav-file-card">
                  <span class="nav-file-icon">{file_icon}</span>
                  <div class="nav-file-info">
                    <span class="nav-file-name">{upload.name}</span>
                    <span class="nav-file-meta">{df.height:,} rows &middot; {df.width} cols</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p class='nav-hint'>CSV &middot; TSV &middot; Parquet &middot; JSON &middot; XLSX &middot; NDJSON</p>",
                unsafe_allow_html=True,
            )

    else:  # Posit Connect
        env_url = os.environ.get("CONNECT_SERVER", "")
        env_key = os.environ.get("CONNECT_API_KEY", "")

        if env_url or env_key:
            shield = _icon("lucide:shield-check")
            st.markdown(
                f'<div class="nav-env-note">{shield} Pre-filled from environment</div>',
                unsafe_allow_html=True,
            )

        use_custom = st.checkbox(
            "Use custom credentials",
            key="sb_use_custom",
        )

        if use_custom:
            server_url = st.text_input(
                "Server URL",
                value=env_url,
                placeholder="https://connect.example.com",
                key="sb_connect_url",
            )
            api_key = st.text_input(
                "API key",
                value=env_key,
                type="password",
                key="sb_connect_key",
            )
            if not (env_url or env_key):
                st.markdown(
                    "<p class='nav-hint'>Set <code>CONNECT_SERVER</code> + "
                    "<code>CONNECT_API_KEY</code> env vars to pre-fill.</p>",
                    unsafe_allow_html=True,
                )
        else:
            server_url = env_url
            api_key = env_key

        connected = st.session_state.get("_connect_connected", False)

        if connected:
            col_status, col_disc = st.columns([3, 2])
            with col_status:
                plug = _icon("lucide:plug-zap")
                st.markdown(
                    f'<div class="nav-connected">{plug} Connected</div>',
                    unsafe_allow_html=True,
                )
            with col_disc:
                if st.button("Disconnect", key="sb_disconnect_btn", icon=":material/link_off:", use_container_width=True):
                    for k in ("_connect_connected", "_connect_pins",
                              "_connect_url", "_connect_key", "_connect_error"):
                        st.session_state.pop(k, None)
                    st.rerun()
        else:
            if st.button(
                "Connect",
                key="sb_connect_btn",
                icon=":material/lan:",
                type="primary",
            ):
                if not server_url or not api_key:
                    st.session_state["_connect_error"] = (
                        "Enter both a Server URL and an API key."
                    )
                else:
                    with st.spinner("Connecting\u2026"):
                        try:
                            pins = list_pins(server_url, api_key)
                            st.session_state["_connect_pins"] = pins
                            st.session_state["_connect_url"] = server_url
                            st.session_state["_connect_key"] = api_key
                            st.session_state["_connect_connected"] = True
                            st.session_state.pop("_connect_error", None)
                            st.rerun()
                        except Exception as exc:
                            st.session_state["_connect_error"] = str(exc)
                            st.session_state.pop("_connect_connected", None)

        if st.session_state.get("_connect_error"):
            st.error(st.session_state["_connect_error"])

        if connected:
            pins = st.session_state.get("_connect_pins", [])
            pin_name = st.selectbox(
                "Select pin",
                options=[""] + pins,
                key="sb_pin_select",
            )
            if pin_name:
                _ALLOWED_EXT = {".csv", ".parquet"}
                try:
                    pin_files = [
                        f for f in get_pin_files(
                            st.session_state["_connect_url"],
                            st.session_state["_connect_key"],
                            pin_name,
                        )
                        if any(f.lower().endswith(ext) for ext in _ALLOWED_EXT)
                    ]
                except Exception:
                    pin_files = []

                if len(pin_files) > 1:
                    # Multiple files — show picker
                    file_name = st.selectbox(
                        "Select file in pin",
                        options=[""] + pin_files,
                        key="sb_pin_file_select",
                    )
                    if file_name:
                        with st.spinner(f"Loading {file_name}…"):
                            try:
                                df = load_pin_file(
                                    st.session_state["_connect_url"],
                                    st.session_state["_connect_key"],
                                    pin_name,
                                    file_name,
                                )
                                source_label = f"pin: {pin_name}/{file_name}"
                            except Exception as exc:
                                st.error(f"Failed to read file: {exc}")
                        if df is not None:
                            file_icon = _icon("lucide:file-text")
                            st.markdown(
                                f"""
                                <div class="nav-file-card">
                                  <span class="nav-file-icon">{file_icon}</span>
                                  <div class="nav-file-info">
                                    <span class="nav-file-name">{file_name}</span>
                                    <span class="nav-file-meta">{df.height:,} rows &middot; {df.width} cols</span>
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                elif len(pin_files) == 1:
                    # Single known file — load directly, no picker needed
                    file_name = pin_files[0]
                    with st.spinner(f"Loading {file_name}…"):
                        try:
                            df = load_pin_file(
                                st.session_state["_connect_url"],
                                st.session_state["_connect_key"],
                                pin_name,
                                file_name,
                            )
                            source_label = f"pin: {pin_name}"
                        except Exception as exc:
                            st.error(f"Failed to read pin: {exc}")
                    if df is not None:
                        file_icon = _icon("lucide:file-text")
                        st.markdown(
                            f"""
                            <div class="nav-file-card">
                              <span class="nav-file-icon">{file_icon}</span>
                              <div class="nav-file-info">
                                <span class="nav-file-name">{file_name}</span>
                                <span class="nav-file-meta">{df.height:,} rows &middot; {df.width} cols</span>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    # No CSV/Parquet files found — fall back to pin_read
                    with st.spinner("Loading pin…"):
                        try:
                            df = load_pin(
                                st.session_state["_connect_url"],
                                st.session_state["_connect_key"],
                                pin_name,
                            )
                            source_label = f"pin: {pin_name}"
                        except Exception as exc:
                            st.error(f"Failed to read pin: {exc}")

    return df, source_label


def _settings(df: pl.DataFrame | None) -> int:
    _section_label("Settings")
    return st.slider(
        "Max rows",
        min_value=1_000,
        max_value=200_000,
        value=20_000,
        step=1_000,
        help="Only this many rows are sent to the browser. The grid is virtualized.",
    )


def _footer(df: pl.DataFrame | None) -> None:
    if df is not None:
        dot_class = "ok"
        meta = (
            f"{df.height:,} rows &middot; {df.width} cols"
            f" &middot; {df.estimated_size('mb'):.1f} MB"
        )
    else:
        dot_class = "idle"
        meta = "No data loaded"
    st.markdown(
        f"""
        <div class="nav-footer">
          <span class="nav-status-dot {dot_class}"></span>
          <span class="nav-footer-meta">{meta}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> SidebarState:
    with st.sidebar:
        _brand()
        st.markdown('<div class="nav-body">', unsafe_allow_html=True)
        df, source_label = _data_source()
        st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
        max_render = _settings(df)
        st.markdown('</div>', unsafe_allow_html=True)
        _footer(df)
    return SidebarState(
        df=df,
        source_label=source_label,
        max_render=max_render,
    )
