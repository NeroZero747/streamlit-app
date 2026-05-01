"""Per-column Polars filter UI driven by full-frame metadata."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import streamlit as st

from .schema import DROPDOWN_MAX_UNIQUE, column_filter_meta


def polars_filter_ui(df: pl.DataFrame) -> pl.DataFrame:
    """
    Custom filter UI driven by the FULL Polars frame.

    - Numeric / date columns: range slider over true min/max of the full frame.
    - Categorical with <= DROPDOWN_MAX_UNIQUE unique values: multiselect with
      ALL distinct values from the full frame.
    - Categorical with > DROPDOWN_MAX_UNIQUE unique values: "contains" text input.

    Returns a filtered Polars frame.
    """
    df_id = f"{id(df)}-{df.height}-{df.width}"
    nonce = st.session_state.get("_filter_nonce", 0)

    with st.expander("🔍 Filters", expanded=True):
        active_key = f"active_filter_cols_{nonce}"
        currently_active = st.session_state.get(active_key, [])

        hdr_l, hdr_r = st.columns([5, 1])
        with hdr_l:
            chosen = st.multiselect(
                "Filter by columns",
                options=df.columns,
                default=[],
                key=active_key,
                help="Pick which columns you want to filter on.",
                label_visibility="collapsed",
                placeholder="Add columns to filter…",
            )
        with hdr_r:
            if currently_active or chosen:
                clear_clicked = st.button(
                    "Clear",
                    key=f"clear_filters_btn_{nonce}",
                    use_container_width=True,
                    help="Reset all active filters.",
                    type="secondary",
                )
                st.markdown(
                    f"""
                    <style>
                      div[data-testid="stButton"]:has(button[kind="secondary"][data-testid*="clear_filters_btn_{nonce}"]) button,
                      .stButton button[data-testid*="clear_filters_btn"] {{
                        background: transparent !important;
                        border: 1px solid var(--border) !important;
                        color: var(--muted) !important;
                        font-size: 0.78rem !important;
                        font-weight: 500 !important;
                        padding: 0.35rem 0.5rem !important;
                        box-shadow: none !important;
                      }}
                      .stButton button[data-testid*="clear_filters_btn"]:hover {{
                        color: var(--text-strong) !important;
                        border-color: var(--border-accent) !important;
                        background: var(--accent-soft) !important;
                      }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                if clear_clicked:
                    for k in [
                        key for key in list(st.session_state.keys())
                        if key.startswith("active_filter_cols_")
                        or key.startswith("flt_")
                    ]:
                        del st.session_state[k]
                    st.session_state["_filter_nonce"] = nonce + 1
                    try:
                        st.rerun(scope="fragment")
                    except TypeError:
                        st.rerun()

        if not chosen:
            st.caption("No filters active — pick a column above to add one.")
            return df

        out = df
        for col in chosen:
            meta = column_filter_meta(df, col, df_id)
            kind = meta["kind"]
            n_unique = meta["n_unique"]

            if kind == "numeric" and "min" in meta and meta["min"] != meta["max"]:
                lo, hi = float(meta["min"]), float(meta["max"])
                lo_sel, hi_sel = st.slider(
                    f"{col}  ·  range",
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                    key=f"flt_num_{col}",
                )
                out = out.filter(
                    (pl.col(col).is_null())
                    | ((pl.col(col) >= lo_sel) & (pl.col(col) <= hi_sel))
                )

            elif kind == "datetime" and "min" in meta and meta["min"] != meta["max"]:
                dmin_raw, dmax_raw = meta["min"], meta["max"]
                dmin = dmin_raw.date() if isinstance(dmin_raw, datetime) else dmin_raw
                dmax = dmax_raw.date() if isinstance(dmax_raw, datetime) else dmax_raw

                if meta.get("parsed_as_date"):
                    col_expr = pl.col(col).str.to_date(strict=False)
                else:
                    col_expr = pl.col(col)

                mode = st.radio(
                    f"{col}  ·  date filter",
                    options=[
                        "Range",
                        "Single date",
                        "Before / On / After",
                        "Last N days",
                        "Quick presets",
                    ],
                    horizontal=True,
                    key=f"flt_dt_mode_{col}",
                )

                if mode == "Range":
                    picked = st.date_input(
                        f"{col}  ·  start \u2192 end",
                        value=(dmin, dmax),
                        min_value=dmin,
                        max_value=dmax,
                        format="MM/DD/YYYY",
                        key=f"flt_dt_range_{col}",
                    )
                    if isinstance(picked, tuple) and len(picked) == 2:
                        start, end = picked
                        out = out.filter(
                            (col_expr.is_null())
                            | ((col_expr >= start) & (col_expr <= end))
                        )

                elif mode == "Single date":
                    picked = st.date_input(
                        f"{col}  ·  on date",
                        value=dmax,
                        min_value=dmin,
                        max_value=dmax,
                        format="MM/DD/YYYY",
                        key=f"flt_dt_single_{col}",
                    )
                    out = out.filter(col_expr == picked)

                elif mode == "Before / On / After":
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        op = st.selectbox(
                            "Op",
                            options=["<=", "<", "==", ">", ">="],
                            index=0,
                            key=f"flt_dt_op_{col}",
                            label_visibility="collapsed",
                        )
                    with c2:
                        picked = st.date_input(
                            f"{col}  ·  vs date",
                            value=dmax,
                            min_value=dmin,
                            max_value=dmax,
                            format="MM/DD/YYYY",
                            key=f"flt_dt_cmp_{col}",
                            label_visibility="collapsed",
                        )
                    expr_map = {
                        "<=": col_expr <= picked,
                        "<": col_expr < picked,
                        "==": col_expr == picked,
                        ">": col_expr > picked,
                        ">=": col_expr >= picked,
                    }
                    out = out.filter(expr_map[op])

                elif mode == "Last N days":
                    n_days = st.number_input(
                        f"{col}  ·  last N days (relative to max date in data)",
                        min_value=1,
                        max_value=max(1, (dmax - dmin).days),
                        value=min(30, max(1, (dmax - dmin).days)),
                        step=1,
                        key=f"flt_dt_lastn_{col}",
                    )
                    cutoff = dmax - timedelta(days=int(n_days))
                    out = out.filter((col_expr >= cutoff) & (col_expr <= dmax))

                else:  # Quick presets
                    today = date.today()
                    presets = {
                        "All": (dmin, dmax),
                        "Today": (today, today),
                        "Yesterday": (
                            today - timedelta(days=1),
                            today - timedelta(days=1),
                        ),
                        "Last 7 days": (today - timedelta(days=7), today),
                        "Last 30 days": (today - timedelta(days=30), today),
                        "Last 90 days": (today - timedelta(days=90), today),
                        "Year to date": (date(today.year, 1, 1), today),
                        "Previous year": (
                            date(today.year - 1, 1, 1),
                            date(today.year - 1, 12, 31),
                        ),
                    }
                    choice = st.selectbox(
                        f"{col}  ·  preset",
                        options=list(presets.keys()),
                        key=f"flt_dt_preset_{col}",
                    )
                    start, end = presets[choice]
                    start = max(start, dmin)
                    end = min(end, dmax)
                    if start <= end:
                        st.caption(
                            f"Filter window: **{start.strftime('%m/%d/%Y')}** → **{end.strftime('%m/%d/%Y')}**"
                        )
                        out = out.filter((col_expr >= start) & (col_expr <= end))
                    else:
                        st.caption(
                            "Preset window is outside the data range — no rows match."
                        )
                        out = out.filter(pl.lit(False))

            else:
                # Categorical
                if n_unique <= DROPDOWN_MAX_UNIQUE and "uniques" in meta:
                    options = meta["uniques"]
                    selected = st.multiselect(
                        f"{col}  ·  values  ({n_unique})",
                        options=options,
                        default=options,
                        key=f"flt_cat_{col}",
                    )
                    if selected and len(selected) < len(options):
                        out = out.filter(pl.col(col).cast(pl.Utf8).is_in(selected))
                else:
                    term = st.text_input(
                        f"{col}  ·  contains  ({n_unique:,} unique — too many for dropdown)",
                        key=f"flt_txt_{col}",
                    )
                    if term:
                        out = out.filter(
                            pl.col(col)
                            .cast(pl.Utf8)
                            .str.to_lowercase()
                            .str.contains(term.lower(), literal=True)
                        )
        return out
