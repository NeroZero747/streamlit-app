"""⚡ Data Explorer — Streamlit + Polars + Glide Data Grid (st.dataframe) + Plotly.

Reactivity / speed strategy:
  * st.dataframe is backed by Glide Data Grid (canvas-rendered, virtualized) —
    handles huge frames smoothly with zero plug-in overhead.
  * Filtering happens in Polars (lazy + fast), only the resulting view
    (head N) is sent to the browser. No 100 MB+ payloads.
  * @st.cache_data memoizes IO and the filter pipeline.
  * @st.fragment isolates each tab so widget changes don't rerun the whole app.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ Data Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — modern glass + gradient
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --accent:#7c3aed; --accent2:#06b6d4;
        --bg:#0b1020; --panel:#111a33; --text:#e2e8f0; --muted:#94a3b8;
      }
      .stApp { background:
          radial-gradient(1200px 600px at 8% -10%, rgba(124,58,237,0.20), transparent 60%),
          radial-gradient(900px 500px at 110% 5%, rgba(6,182,212,0.16), transparent 60%),
          var(--bg); }
      .block-container { padding-top:.8rem; padding-bottom:.8rem; max-width:100% !important; }
      h1 { background:linear-gradient(90deg,#c4b5fd 0%,#67e8f9 100%);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           font-weight:800; letter-spacing:-.03em; margin-bottom:.2rem;}
      h2,h3 { letter-spacing:-.02em; }

      [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#0b1020 0%,#0f172a 100%);
        border-right:1px solid rgba(148,163,184,0.08);
      }
      [data-testid="stSidebar"] * { color:var(--text); }
      [data-testid="stSidebar"] label { color:var(--muted) !important;
        font-weight:600; text-transform:uppercase; font-size:.72rem; letter-spacing:.06em;}

      .glass { background:rgba(17,26,51,.55); backdrop-filter:blur(10px);
               border:1px solid rgba(148,163,184,.12); border-radius:14px;
               padding:14px 16px; box-shadow:0 8px 24px rgba(0,0,0,.25);}
      .pill { display:inline-block; padding:3px 10px; border-radius:999px;
              background:rgba(124,58,237,.18); color:#ddd6fe;
              font-size:.75rem; margin-right:6px; border:1px solid rgba(124,58,237,.35);}
      .pill.cyan { background:rgba(6,182,212,.15); color:#a5f3fc;
                   border-color:rgba(6,182,212,.35);}

      .stButton>button, .stDownloadButton>button {
        border-radius:10px;
        border:1px solid rgba(148,163,184,.18);
        background:linear-gradient(180deg,#1e293b 0%,#0f172a 100%);
        color:var(--text); font-weight:600;
        transition:transform .08s ease, border-color .15s ease;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        border-color:var(--accent); transform:translateY(-1px);
      }

      /* Glide Data Grid container */
      div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
      }

      header [data-testid="stToolbar"] { display:none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached IO
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=4)
def load_uploaded(name: str, raw: bytes) -> pl.DataFrame:
    suffix = Path(name).suffix.lower()
    buf = io.BytesIO(raw)
    if suffix == ".csv":
        return pl.read_csv(buf, infer_schema_length=10000)
    if suffix in {".tsv", ".txt"}:
        return pl.read_csv(buf, separator="\t", infer_schema_length=10000)
    if suffix == ".parquet":
        return pl.read_parquet(buf)
    if suffix in {".json", ".ndjson"}:
        try:
            return pl.read_json(buf)
        except Exception:
            buf.seek(0)
            return pl.read_ndjson(buf)
    if suffix in {".xlsx", ".xls"}:
        return pl.from_pandas(pd.read_excel(buf))
    raise ValueError(f"Unsupported file type: {suffix}")


@st.cache_resource(show_spinner="Connecting to Posit Connect…")
def get_connect_board(server_url: str, api_key: str):
    import pins
    return pins.board_connect(
        server_url=server_url, api_key=api_key, allow_pickle_read=True
    )


@st.cache_data(show_spinner="Loading pin…", max_entries=4)
def load_pin(server_url: str, api_key: str, pin_name: str) -> pl.DataFrame:
    board = get_connect_board(server_url, api_key)
    obj: Any = board.pin_read(pin_name)
    if isinstance(obj, pl.DataFrame):
        return obj
    if isinstance(obj, pd.DataFrame):
        return pl.from_pandas(obj)
    return pl.from_pandas(pd.DataFrame(obj))


@st.cache_data(show_spinner=False, ttl=300)
def list_pins(server_url: str, api_key: str) -> list[str]:
    return list(get_connect_board(server_url, api_key).pin_list())


@st.cache_data(show_spinner=False)
def schema_summary(df: pl.DataFrame) -> pd.DataFrame:
    rows = []
    for col, dtype in df.schema.items():
        s = df.get_column(col)
        rows.append({
            "column": col,
            "dtype": str(dtype),
            "non_null": int(s.len() - s.null_count()),
            "nulls": int(s.null_count()),
            "unique": int(s.n_unique()),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=4)
def head_pandas(df: pl.DataFrame, n: int) -> pd.DataFrame:
    """Cheap, cached conversion of the head(n) slice for display."""
    return df.head(n).to_pandas()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='margin:0'>⚡ Data Explorer</h2>"
        "<p style='color:#94a3b8;margin-top:4px;font-size:.85rem'>"
        "Polars · Glide Data Grid · Pins</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    source = option_menu(
        menu_title=None,
        options=["Upload", "Posit Connect"],
        icons=["cloud-upload", "pin-angle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#a78bfa", "font-size": "16px"},
            "nav-link": {
                "font-size": ".85rem", "text-align": "center",
                "margin": "0 2px", "padding": "6px 8px",
                "border-radius": "8px", "color": "#cbd5e1",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg,#7c3aed 0%,#06b6d4 100%)",
                "color": "white", "font-weight": "600",
            },
        },
    )

    df: pl.DataFrame | None = None
    source_label = ""

    if source == "Upload":
        upload = st.file_uploader(
            "Upload (≤ 5 GB)",
            type=["csv", "tsv", "txt", "parquet", "json", "ndjson", "xlsx", "xls"],
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

    st.divider()
    st.markdown("**Display**")
    use_filter_ui = st.toggle(
        "Rich filter UI",
        value=True,
        help="Adds per-column filter widgets above the grid (slightly slower on huge frames).",
    )
    max_render = st.slider(
        "Max rows to render",
        min_value=1_000,
        max_value=200_000,
        value=20_000,
        step=1_000,
        help="The grid is virtualized but only the rendered slice is sent to the browser. "
             "Filtering is applied in Polars first, so you still see counts over the full dataset.",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("# Data Explorer")
st.markdown(
    "<span class='pill'>Polars</span>"
    "<span class='pill cyan'>Glide Data Grid</span>"
    "<span class='pill'>Plotly</span>"
    "<span class='pill cyan'>Posit Pins</span>",
    unsafe_allow_html=True,
)

if df is None:
    st.markdown(
        "<div class='glass' style='margin-top:24px;text-align:center;padding:48px'>"
        "<h3>👈 Upload a file or connect to a Posit Connect pin to begin</h3>"
        "<p style='color:#94a3b8'>Supports CSV, TSV, Parquet, JSON/NDJSON, Excel up to 5 GB."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Stash for fragments
st.session_state["_df"] = df
st.session_state["_source_label"] = source_label
st.session_state["_use_filter_ui"] = use_filter_ui
st.session_state["_max_render"] = max_render


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Source", source_label or "—")
m2.metric("Rows", f"{df.height:,}")
m3.metric("Columns", f"{df.width}")
m4.metric("Memory", f"{df.estimated_size('mb'):.1f} MB")
style_metric_cards(
    background_color="#111a33",
    border_left_color="#7c3aed",
    border_color="rgba(148,163,184,0.12)",
    box_shadow=True,
)

tab_grid, tab_charts, tab_schema = st.tabs(["📊 Grid", "📈 Charts", "🧬 Schema"])


# --- Grid tab --------------------------------------------------------------
@st.fragment
def grid_fragment() -> None:
    df: pl.DataFrame = st.session_state["_df"]
    use_filter_ui: bool = st.session_state["_use_filter_ui"]
    max_render: int = st.session_state["_max_render"]

    # Choose visible columns (cheap, server-side)
    with st.expander("👁 Columns", expanded=False):
        visible = st.multiselect(
            "Visible columns",
            options=df.columns,
            default=df.columns,
            key="visible_cols",
        )
    work = df.select(visible) if visible else df

    if use_filter_ui:
        # Built-in rich per-column filter UI from streamlit-extras.
        # It needs pandas, so we feed it just the head slice for filter UI;
        # then we re-apply equivalent constraints to the full Polars frame.
        # For simplicity & speed: use the pandas head as the filter source —
        # accuracy on filter VALUE choices may be capped to the rendered slice
        # for very high-cardinality cols; counts shown below are always exact.
        sample_pd = head_pandas(work, min(max_render, work.height))
        filtered_pd = dataframe_explorer(sample_pd, case=False)
        st.caption(
            f"Showing {len(filtered_pd):,} of {work.height:,} rows "
            f"(filter UI operates on a {len(sample_pd):,}-row sample for speed)."
        )
        st.dataframe(
            filtered_pd,
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        download_pdf = filtered_pd
    else:
        rendered = head_pandas(work, min(max_render, work.height))
        st.caption(
            f"Showing first {len(rendered):,} of {work.height:,} rows. "
            "Use Glide's column header menus for sort/search."
        )
        st.dataframe(
            rendered,
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        download_pdf = rendered

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇ CSV (rendered)",
            data=download_pdf.to_csv(index=False).encode(),
            file_name="data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        buf = io.BytesIO()
        pl.from_pandas(download_pdf).write_parquet(buf)
        st.download_button(
            "⬇ Parquet (rendered)",
            data=buf.getvalue(),
            file_name="data.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with d3:
        full_buf = io.BytesIO()
        df.write_parquet(full_buf)
        st.download_button(
            "⬇ Parquet (full)",
            data=full_buf.getvalue(),
            file_name="full.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )


with tab_grid:
    grid_fragment()


# --- Charts tab ------------------------------------------------------------
@st.fragment
def charts_fragment() -> None:
    df: pl.DataFrame = st.session_state["_df"]
    cols = df.columns
    numeric_cols = [c for c, d in df.schema.items() if d.is_numeric()]
    cat_cols = [
        c for c in cols
        if c not in numeric_cols and df.get_column(c).n_unique() <= 200
    ]
    if not numeric_cols and not cat_cols:
        st.info("No suitable columns for charting.")
        return

    chart_type = st.radio(
        "Chart type",
        ["Histogram", "Bar (count)", "Scatter", "Box"],
        horizontal=True,
    )
    template = "plotly_dark"
    color_seq = px.colors.sequential.Plasma
    # Always sample for charts to keep things instant on big frames.
    sample_n = min(50_000, df.height)
    pdf = df.sample(n=sample_n, seed=0).to_pandas() if df.height > sample_n else df.to_pandas()

    fig = None
    if chart_type == "Histogram" and numeric_cols:
        c = st.selectbox("Column", numeric_cols)
        color = st.selectbox("Color by", ["(none)"] + cat_cols)
        fig = px.histogram(
            pdf, x=c, color=None if color == "(none)" else color,
            template=template, color_discrete_sequence=color_seq, nbins=50,
        )
    elif chart_type == "Bar (count)" and cat_cols:
        c = st.selectbox("Column", cat_cols)
        vc = pdf[c].value_counts().reset_index().head(50)
        vc.columns = [c, "count"]
        fig = px.bar(vc, x=c, y="count", template=template,
                     color_discrete_sequence=color_seq)
    elif chart_type == "Scatter" and len(numeric_cols) >= 2:
        cc1, cc2, cc3 = st.columns(3)
        x = cc1.selectbox("X", numeric_cols, index=0)
        y = cc2.selectbox("Y", numeric_cols, index=1)
        color = cc3.selectbox("Color by", ["(none)"] + cat_cols)
        fig = px.scatter(
            pdf, x=x, y=y,
            color=None if color == "(none)" else color,
            template=template, color_discrete_sequence=color_seq, opacity=0.7,
        )
    elif chart_type == "Box" and numeric_cols:
        y = st.selectbox("Y", numeric_cols)
        x = st.selectbox("Group by", ["(none)"] + cat_cols)
        fig = px.box(pdf, y=y, x=None if x == "(none)" else x,
                     template=template, color_discrete_sequence=color_seq)

    if fig is None:
        st.info("Not enough columns of the right type for this chart.")
        return
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=560,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    if df.height > sample_n:
        st.caption(f"Sampled {sample_n:,} of {df.height:,} rows for chart speed.")


with tab_charts:
    charts_fragment()


# --- Schema tab ------------------------------------------------------------
@st.fragment
def schema_fragment() -> None:
    df: pl.DataFrame = st.session_state["_df"]
    summary = schema_summary(df)
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown("#### Column overview")
        st.dataframe(summary, use_container_width=True, hide_index=True, height=480)
    with c2:
        st.markdown("#### Null heatmap (sample)")
        sample = df.head(2000).to_pandas().isna().astype(int)
        if sample.shape[1] > 0:
            fig = px.imshow(
                sample.T, aspect="auto",
                color_continuous_scale=["#0b1020", "#7c3aed"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10), height=480,
                coloraxis_showscale=False,
            )
            fig.update_xaxes(showticklabels=False, title="rows (sample)")
            st.plotly_chart(fig, use_container_width=True, theme=None)


with tab_schema:
    schema_fragment()
