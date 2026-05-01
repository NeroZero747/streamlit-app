"""Grid tab: filtered/visible columns + downloads."""
from __future__ import annotations

import io

import polars as pl
import streamlit as st

from ..downloads import (
    ICON_ARCHIVE,
    ICON_CSV,
    ICON_PARQUET,
    save_as_button,
)
from ..filters import polars_filter_ui
from ..schema import head_pandas


@st.fragment
def grid_fragment() -> None:
    df: pl.DataFrame = st.session_state["_df"]
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

    filtered_pl = polars_filter_ui(work)
    rendered = head_pandas(filtered_pl, min(max_render, filtered_pl.height))
    st.caption(
        f"Previewing first {len(rendered):,} of {filtered_pl.height:,} filtered rows "
        f"(out of {work.height:,} total). Downloads export the full filtered set."
    )
    st.dataframe(rendered, use_container_width=True, hide_index=True, height=620)
    export_pl = filtered_pl

    # Cache built download payloads against a cheap signature of the export
    # frame so we don't re-serialize CSV/Parquet on every filter change.
    export_sig = (
        f"{id(df)}-{export_pl.height}-{export_pl.width}-{hash(tuple(export_pl.columns))}"
    )
    if st.session_state.get("_export_sig") != export_sig:
        for k in ("dl_csv", "dl_parquet"):
            st.session_state.pop(k, None)
        st.session_state["_export_sig"] = export_sig

    def _csv_bytes() -> bytes:
        if "dl_csv" not in st.session_state:
            st.session_state["dl_csv"] = export_pl.write_csv().encode()
        return st.session_state["dl_csv"]

    def _parquet_bytes() -> bytes:
        if "dl_parquet" not in st.session_state:
            buf = io.BytesIO()
            export_pl.write_parquet(buf)
            st.session_state["dl_parquet"] = buf.getvalue()
        return st.session_state["dl_parquet"]

    def _full_parquet_bytes() -> bytes:
        if "dl_full_parquet" not in st.session_state:
            full_buf = io.BytesIO()
            df.write_parquet(full_buf)
            st.session_state["dl_full_parquet"] = full_buf.getvalue()
        return st.session_state["dl_full_parquet"]

    base_name = "data"

    d1, d2, d3 = st.columns(3)
    with d1:
        save_as_button(
            label="Save as CSV",
            sublabel="Filtered · text/csv",
            icon_svg=ICON_CSV,
            accent="#cb187d",
            payload_fn=_csv_bytes,
            suggested_name=f"{base_name}.csv",
            mime="text/csv",
            ext=".csv",
            accept_desc="CSV file",
            component_key="csv",
        )
    with d2:
        save_as_button(
            label="Save as Parquet",
            sublabel="Filtered · columnar",
            icon_svg=ICON_PARQUET,
            accent="#e879f9",
            payload_fn=_parquet_bytes,
            suggested_name=f"{base_name}.parquet",
            mime="application/octet-stream",
            ext=".parquet",
            accept_desc="Parquet file",
            component_key="parquet",
        )
    with d3:
        save_as_button(
            label="Save full Parquet",
            sublabel="Unfiltered · all rows",
            icon_svg=ICON_ARCHIVE,
            accent="#a855f7",
            payload_fn=_full_parquet_bytes,
            suggested_name=f"{base_name}_full.parquet",
            mime="application/octet-stream",
            ext=".parquet",
            accept_desc="Parquet file",
            component_key="full_parquet",
        )
