"""Pivot table tab."""
from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st
from streamlit_pivot import st_pivot_table


@st.fragment
def charts_fragment() -> None:
    df: pl.DataFrame = st.session_state["_df"]
    cols = df.columns
    numeric_cols = [c for c, d in df.schema.items() if d.is_numeric()]
    cat_cols = [
        c for c in cols if c not in numeric_cols and df.get_column(c).n_unique() <= 200
    ]
    if not numeric_cols and not cat_cols:
        st.info("No suitable columns for the pivot table.")
        return

    _render_pivot(df, numeric_cols, cat_cols)


# ----------------------------------------------------------------------
# Pivot Table renderer
# ----------------------------------------------------------------------

_AGG_OPTIONS = [
    "sum", "mean", "median", "min", "max", "count",
    "count_distinct", "first", "last", "std", "var",
]

# Carefully tuned palettes — each picks colors that read clearly on the dark
# Streamlit theme used by this app, with mid-tones that don't wash out text.
_CF_PALETTES: dict[str, dict[str, str]] = {
    "Aurora (indigo → magenta → amber)": {
        "min_color": "#1e1b4b", "mid_color": "#c026d3", "max_color": "#fbbf24",
    },
    "Ocean (deep blue → teal → mint)": {
        "min_color": "#0c4a6e", "mid_color": "#0891b2", "max_color": "#a7f3d0",
    },
    "Forest (slate → emerald → lime)": {
        "min_color": "#0f172a", "mid_color": "#059669", "max_color": "#bef264",
    },
    "Sunset (navy → coral → gold)": {
        "min_color": "#1e293b", "mid_color": "#fb7185", "max_color": "#fde047",
    },
    "Diverging Cool → Warm": {
        "min_color": "#1e3a8a", "mid_color": "#e2e8f0", "max_color": "#b91c1c",
    },
    "Diverging Green ↔ Red": {
        "min_color": "#15803d", "mid_color": "#fef3c7", "max_color": "#b91c1c",
    },
    "Mono Violet": {
        "min_color": "#1e1b4b", "mid_color": "#7c3aed", "max_color": "#e9d5ff",
    },
    "Mono Cyan": {
        "min_color": "#082f49", "mid_color": "#0ea5e9", "max_color": "#cffafe",
    },
}

_DATA_BAR_PRESETS: dict[str, str] = {
    "Violet": "#7c3aed",
    "Cyan": "#06b6d4",
    "Emerald": "#10b981",
    "Amber": "#f59e0b",
    "Rose": "#f43f5e",
    "Slate": "#64748b",
    "Custom": "",
}

_THRESHOLD_PRESETS: dict[str, dict[str, str]] = {
    "Success (green)": {"bg": "#064e3b", "fg": "#a7f3d0"},
    "Warning (amber)": {"bg": "#78350f", "fg": "#fde68a"},
    "Danger (red)":   {"bg": "#7f1d1d", "fg": "#fecaca"},
    "Info (blue)":    {"bg": "#1e3a8a", "fg": "#bfdbfe"},
    "Neutral":        {"bg": "#1f2937", "fg": "#e5e7eb"},
    "Custom":         {"bg": "", "fg": ""},
}

_PAGINATION_CSS = """
<style>
/* Pagination pill buttons */
.st-key-_pv_drill_prev button,
.st-key-_pv_drill_next button {
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.35rem 1.1rem !important;
    background: transparent !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    color: rgba(165,180,252,0.75) !important;
    transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease, box-shadow 0.14s ease !important;
    box-shadow: none !important;
}
.st-key-_pv_drill_prev button:hover:not(:disabled),
.st-key-_pv_drill_next button:hover:not(:disabled) {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.65) !important;
    color: rgb(199,210,254) !important;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.25) !important;
}
.st-key-_pv_drill_prev button:active:not(:disabled),
.st-key-_pv_drill_next button:active:not(:disabled) {
    background: rgba(99,102,241,0.2) !important;
}
.st-key-_pv_drill_prev button:disabled,
.st-key-_pv_drill_next button:disabled {
    opacity: 0.22 !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.3) !important;
}
</style>
"""

def _drill_prev_page() -> None:
    """on_click callback: decrement drill-down page before fragment rerenders."""
    n = st.session_state.get("_pv_drill_page_num", 0)
    if n > 0:
        st.session_state["_pv_drill_page_num"] = n - 1


def _drill_next_page() -> None:
    """on_click callback: increment drill-down page before fragment rerenders."""
    st.session_state["_pv_drill_page_num"] = st.session_state.get("_pv_drill_page_num", 0) + 1


# Polars aggregation expressions — one per supported function.
_PL_AGG: dict[str, Any] = {
    "sum":            lambda c: pl.col(c).sum(),
    "mean":           lambda c: pl.col(c).mean(),
    "median":         lambda c: pl.col(c).median(),
    "min":            lambda c: pl.col(c).min(),
    "max":            lambda c: pl.col(c).max(),
    "count":          lambda c: pl.col(c).count(),
    "count_distinct": lambda c: pl.col(c).n_unique(),
    "first":          lambda c: pl.col(c).first(),
    "last":           lambda c: pl.col(c).last(),
    "std":            lambda c: pl.col(c).std(),
    "var":            lambda c: pl.col(c).var(),
}

_SKIP_TYPES = (pl.List, pl.Struct, pl.Binary, pl.Array)


@st.cache_data(show_spinner=False, max_entries=64)
def _pivot_pandas(
    df_id: int,
    _df: pl.DataFrame,
    dim_cols: tuple[str, ...],
    value_cols: tuple[str, ...],
    agg_map: tuple[tuple[str, str], ...],  # ((field, fn), ...)
) -> "object":  # pandas DataFrame
    """Pre-aggregate with polars then return pandas for st_pivot_table.

    Cached by (df_id, dims, values, agg) — identical configs are instant.
    Result is tiny (cartesian product of dim cardinalities) so the component
    renders it purely in the browser with execution_mode='client_only'.
    """
    agg_dict = dict(agg_map)
    all_usable = [c for c, d in _df.schema.items()
                  if not isinstance(d, _SKIP_TYPES)]

    if not dim_cols or not value_cols:
        return _df.select(all_usable).to_pandas()

    avail_dims = [c for c in dim_cols if c in _df.columns]
    avail_vals = [c for c in value_cols if c in _df.columns]

    if not avail_dims or not avail_vals:
        return _df.select(all_usable).to_pandas()

    exprs = [
        _PL_AGG.get(agg_dict.get(v, "sum"), _PL_AGG["sum"])(v).alias(v)
        for v in avail_vals
    ]
    return (
        _df.select(avail_dims + avail_vals)
           .group_by(avail_dims)
           .agg(exprs)
           .to_pandas()
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _drill_pandas(
    df_id: int,
    _df: pl.DataFrame,
    filter_cols: tuple[str, ...],
    filter_vals: tuple[tuple[str, ...], ...],  # parallel to filter_cols
    offset: int,
    limit: int,
) -> "object":  # pandas DataFrame
    """Filter source rows for drill-down and return pandas (cached)."""
    result = _df
    for col, vals in zip(filter_cols, filter_vals):
        result = result.filter(pl.col(col).is_in(list(vals)))
    return result.slice(offset, limit).to_pandas()


@st.fragment
def _drill_fragment(df: pl.DataFrame, dim_cols_all: list[str]) -> None:
    """Paginated drill-down — runs as a fragment so Prev/Next don't rerun the pivot."""
    if not dim_cols_all:
        st.caption("Add at least one Rows or Columns field to enable drill-down.")
        return

    page_size = st.select_slider(
        "Rows per page", options=[100, 250, 500, 1000, 2500, 5000], value=500,
        key="_pv_drill_page",
    )
    drill_cols_ui = st.columns(min(4, len(dim_cols_all)))
    filter_cols: list[str] = []
    filter_vals: list[tuple] = []
    for i, dim in enumerate(dim_cols_all):
        unique_vals = sorted(df[dim].drop_nulls().unique().to_list())
        sel = drill_cols_ui[i % len(drill_cols_ui)].multiselect(
            dim, unique_vals, key=f"_pv_drill_{dim}", placeholder="All values",
        )
        if sel:
            filter_cols.append(dim)
            filter_vals.append(tuple(sel))

    if not filter_cols:
        st.caption("Select dimension values above to filter source records.")
        return

    mask = df
    for col, vals in zip(filter_cols, filter_vals):
        mask = mask.filter(pl.col(col).is_in(list(vals)))
    total_matches = mask.height
    total_pages = max(1, -(-total_matches // page_size))

    # Reset to page 0 when filters or page size change
    _sig = str((filter_cols, filter_vals, page_size))
    if st.session_state.get("_pv_drill_state_sig") != _sig:
        st.session_state["_pv_drill_state_sig"] = _sig
        st.session_state["_pv_drill_page_num"] = 0

    page_num: int = max(0, min(st.session_state.get("_pv_drill_page_num", 0), total_pages - 1))
    offset = page_num * page_size
    row_from, row_to = offset + 1, min(offset + page_size, total_matches)

    st.caption(
        f"{total_matches:,} rows match \u2014 "
        f"showing {row_from:,}\u2013{row_to:,}  ·  page {page_num + 1} of {total_pages:,}"
    )

    if total_matches > 0:
        drill_pdf = _drill_pandas(
            id(df), df, tuple(filter_cols), tuple(filter_vals), offset, page_size,
        )
        st.dataframe(drill_pdf, use_container_width=True, hide_index=True, height=380)

        st.markdown(_PAGINATION_CSS, unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns([1, 3, 1])
        pc1.button(
            "\u2039\u2039  Previous", disabled=page_num == 0,
            on_click=_drill_prev_page,
            key="_pv_drill_prev", use_container_width=True,
        )
        pc2.markdown(
            f"<p style='text-align:center;margin:9px 0 4px;opacity:.45;"
            f"font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;'>"
            f"Page&nbsp;&nbsp;{page_num + 1}&nbsp;&nbsp;of&nbsp;&nbsp;{total_pages:,}</p>",
            unsafe_allow_html=True,
        )
        pc3.button(
            "Next  \u203a\u203a", disabled=page_num >= total_pages - 1,
            on_click=_drill_next_page,
            key="_pv_drill_next", use_container_width=True,
        )


def _render_pivot(df: pl.DataFrame, numeric_cols: list[str], cat_cols: list[str]) -> None:
    """Reactive pivot table: polars pre-aggregation + client-only rendering.

    Speed strategy
    ──────────────
    1. Pre-aggregate with polars (fast) → cached pandas (tiny, e.g. 12 rows)
    2. Pass tiny frame to st_pivot_table with execution_mode='client_only'
       → browser renders immediately, zero server round-trips on every change.
    3. Custom paginated drill-down uses polars filter + st.dataframe instead
       of the component's threshold_hybrid server round-trip.
    """
    non_numeric = [c for c, d in df.schema.items() if not d.is_numeric()]
    all_cols = df.columns

    # ── Reset all pivot state when the dataset changes ────────────────
    # If the user loads a different file, column names / types change.
    # Stale _pv_rows / _pv_cols values that no longer exist in the new
    # dataset are silently dropped by the multiselect, producing an empty
    # selection → python_config_changed=False → the component falls back
    # to its own cached config from the previous dataset. Wiping all
    # _pv_* keys forces fresh defaults and a clean component state.
    _df_sig = (df.height, df.width, tuple(df.columns))
    if st.session_state.get("_pv_df_sig") != _df_sig:
        st.session_state["_pv_df_sig"] = _df_sig
        for _k in list(st.session_state.keys()):
            if _k.startswith("_pv_") or _k == "data_explorer_pivot":
                del st.session_state[_k]
        st.session_state["_pv_df_sig"] = _df_sig  # restore after wipe

    # ════════════════════════════════════════════════════════════════════
    # 1 · FIELD CONFIGURATION
    #     Choose which columns appear on Rows, Columns, Values & Filters.
    #     Changes apply instantly — no submit button needed.
    # ════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='pv-sec-row'>"
        "<div class='pv-sec-bar'></div>"
        "<span class='pv-sec-label'>1 &nbsp;·&nbsp; Field Configuration</span>"
        "<span class='pv-sec-desc'>— choose dimensions, measures and filters</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    fc1, fc2, fc3, fc4 = st.columns(4)
    rows_sel: list[str] = fc1.multiselect(
        "Rows", all_cols, default=cat_cols[:1], key="_pv_rows",
        help="Dimensions to group rows by.",
    )
    cols_sel: list[str] = fc2.multiselect(
        "Columns", all_cols,
        default=cat_cols[1:2] if len(cat_cols) > 1 else [],
        key="_pv_cols",
        help="Dimensions to pivot across columns.",
    )
    vals_sel: list[str] = fc3.multiselect(
        "Values", numeric_cols, default=numeric_cols[:1], key="_pv_vals",
        help="Numeric fields to aggregate.",
    )
    ff_sel: list[str] = fc4.multiselect(
        "Filters", all_cols, default=[], key="_pv_ff",
        help="Report-level filter chips shown inside the pivot header.",
    )

    # Per-value aggregation
    agg_sel: dict[str, str] = {}
    if vals_sel:
        st.markdown(
            "<p style='font-size:.72rem;opacity:.5;margin:6px 0 2px;'>"
            "Aggregation per value field</p>",
            unsafe_allow_html=True,
        )
        n_agg = min(4, len(vals_sel))
        agg_ui = st.columns(n_agg)
        for i, v in enumerate(vals_sel):
            agg_sel[v] = agg_ui[i % n_agg].selectbox(
                v, _AGG_OPTIONS, index=0, key=f"_pv_agg_{v}"
            )

    st.divider()

    # ════════════════════════════════════════════════════════════════════
    # 2 · DISPLAY OPTIONS
    #     Totals, layout, number format, and conditional formatting.
    # ════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='pv-sec-row'>"
        "<div class='pv-sec-bar'></div>"
        "<span class='pv-sec-label'>2 &nbsp;·&nbsp; Display Options</span>"
        "<span class='pv-sec-desc'>— table layout, totals and conditional formatting</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    adv_col, cf_col = st.columns(2)

    with adv_col:
        with st.expander("⚙  Table options", expanded=False):
            r1, r2 = st.columns(2)
            show_totals       = r1.checkbox("Grand totals",  value=True,  key="_pv_totals")
            show_subtotals    = r2.checkbox("Subtotals",     value=False, key="_pv_subt")
            r3, r4 = st.columns(2)
            sticky_headers    = r3.checkbox("Sticky headers", value=True,  key="_pv_sticky")
            repeat_row_labels = r4.checkbox("Repeat labels",  value=False, key="_pv_rep")
            row_layout = st.radio(
                "Row layout", ["table", "hierarchy"], horizontal=True, key="_pv_layout"
            )
            auto_date_hierarchy = st.checkbox("Auto date hierarchy", value=True, key="_pv_dateh")
            r7, r8 = st.columns(2)
            number_format = r7.text_input(
                "Number format", value=",.2f", key="_pv_nf",
                help="d3 format string, e.g. ',.2f' or '$,.0f'",
            )
            empty_cell_value = r8.text_input("Empty cell", value="-", key="_pv_empty")

    with cf_col:
        with st.expander("🎨  Conditional formatting", expanded=False):
            with st.form("_pv_cf_form"):
                cf_mode = st.radio(
                    "Style",
                    ["Color scale", "Data bars", "Threshold", "None"],
                    horizontal=True, key="_pv_cf_mode",
                )
                cf_fields_sel = st.multiselect(
                    "Apply to (empty = all numeric)",
                    options=numeric_cols, default=[], key="_pv_cf_fields",
                )
                cf_scope = st.radio(
                    "Scope", ["per_column", "global"], horizontal=True,
                    key="_pv_cf_scope",
                )
                if cf_mode == "Color scale":
                    pal_c, mid_c = st.columns([2, 1])
                    palette_name = pal_c.selectbox(
                        "Palette", list(_CF_PALETTES.keys()), key="_pv_cf_pal"
                    )
                    use_mid = mid_c.checkbox("Anchor midpoint", value=False, key="_pv_cf_use_mid")
                    _p = _CF_PALETTES[palette_name]
                    st.markdown(
                        f"<div style='height:12px;border-radius:4px;margin:2px 0 6px;"
                        f"background:linear-gradient(to right,{_p['min_color']},"
                        f"{_p['mid_color']},{_p['max_color']});'></div>",
                        unsafe_allow_html=True,
                    )
                    mid_value = st.number_input("Midpoint", value=0.0, key="_pv_cf_mid", disabled=not use_mid)
                elif cf_mode == "Data bars":
                    bar_c, pick_c = st.columns(2)
                    bar_preset = bar_c.selectbox("Bar color", list(_DATA_BAR_PRESETS.keys()), key="_pv_cf_bar")
                    bar_custom = pick_c.color_picker("Custom", value="#7c3aed", key="_pv_cf_bar_custom", disabled=bar_preset != "Custom")
                elif cf_mode == "Threshold":
                    op_c, v1_c, v2_c = st.columns(3)
                    thr_op   = op_c.selectbox("Operator", ["gt", "gte", "lt", "lte", "eq", "between"], key="_pv_cf_op")
                    thr_val  = v1_c.number_input("Value",           value=0.0, key="_pv_cf_thr")
                    thr_val2 = v2_c.number_input("Upper (between)", value=0.0, key="_pv_cf_thr2")
                    theme_name = st.selectbox("Color theme", list(_THRESHOLD_PRESETS.keys()), key="_pv_cf_theme")
                    bg_c, fg_c = st.columns(2)
                    thr_bg = bg_c.color_picker("Background", value="#fde68a", key="_pv_cf_bg")
                    thr_fg = fg_c.color_picker("Text color",  value="#e5e7eb", key="_pv_cf_fg")
                cf_submitted = st.form_submit_button("Apply formatting", type="primary", use_container_width=True)

            if cf_submitted:
                apply_to = cf_fields_sel if cf_fields_sel else list(numeric_cols)
                cf_rules: list[dict] = []
                if cf_mode == "Color scale" and apply_to:
                    pal = _CF_PALETTES[palette_name]
                    rule: dict = {
                        "type": "color_scale", "apply_to": apply_to, "scope": cf_scope,
                        "min_color": pal["min_color"], "mid_color": pal["mid_color"], "max_color": pal["max_color"],
                    }
                    if use_mid:
                        rule["mid_value"] = float(mid_value)
                    cf_rules.append(rule)
                elif cf_mode == "Data bars" and apply_to:
                    color = bar_custom if bar_preset == "Custom" else _DATA_BAR_PRESETS[bar_preset]
                    cf_rules.append({"type": "data_bars", "apply_to": apply_to, "scope": cf_scope, "color": color})
                elif cf_mode == "Threshold" and apply_to:
                    theme = _THRESHOLD_PRESETS[theme_name]
                    bg = thr_bg if theme_name == "Custom" else theme["bg"]
                    fg = thr_fg if theme_name == "Custom" else theme["fg"]
                    cond: dict = {"operator": thr_op, "value": float(thr_val), "background_color": bg, "text_color": fg}
                    if thr_op == "between":
                        cond["value2"] = float(thr_val2)
                    cf_rules.append({"type": "threshold", "apply_to": apply_to, "conditions": [cond]})
                st.session_state["_pv_cf_cfg"] = cf_rules or None

    st.divider()

    # ════════════════════════════════════════════════════════════════════
    # 3 · PIVOT TABLE
    # ════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='pv-sec-row'>"
        "<div class='pv-sec-bar'></div>"
        "<span class='pv-sec-label'>3 &nbsp;·&nbsp; Pivot Table</span>"
        "<span class='pv-sec-desc'>— click any cell to drill down into source records</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    dim_tuple = tuple(dict.fromkeys(rows_sel + cols_sel))
    val_tuple = tuple(vals_sel)
    agg_tuple = tuple(sorted(agg_sel.items())) if agg_sel else ()

    pdf = _pivot_pandas(id(df), df, dim_tuple, val_tuple, agg_tuple)

    st_pivot_table(
        pdf,
        key="data_explorer_pivot",
        rows=rows_sel or None,
        columns=cols_sel or None,
        values=vals_sel or None,
        aggregation=agg_sel if agg_sel else "sum",
        filter_fields=ff_sel or None,
        show_totals=show_totals,
        show_subtotals=show_subtotals,
        sticky_headers=sticky_headers,
        repeat_row_labels=repeat_row_labels,
        row_layout=row_layout,
        auto_date_hierarchy=auto_date_hierarchy,
        enable_drilldown=True,
        on_cell_click=lambda: None,
        number_format=number_format.strip() or None,
        empty_cell_value=empty_cell_value or "-",
        conditional_formatting=st.session_state.get("_pv_cf_cfg"),
        hidden_from_aggregators=non_numeric or None,
        show_sections=False,
        interactive=True,
        max_height=620,
        export_filename="pivot-table",
        execution_mode="client_only",
        style="striped",
        menu_limit=500,
    )

    st.caption(
        f"{df.height:,} rows · {len(df.columns)} cols · "
        f"pivot pre-aggregated to {len(pdf):,} rows"
    )

    # ════════════════════════════════════════════════════════════════════
    # 4 · DRILL-DOWN
    #     Click any pivot cell to auto-filter, or pick values manually.
    #     Results are paginated and shown in a source-record grid below.
    # ════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div class='pv-sec-row'>"
        "<div class='pv-sec-bar'></div>"
        "<span class='pv-sec-label'>4 &nbsp;·&nbsp; Drill-Down &nbsp;·&nbsp; Source Records</span>"
        "<span class='pv-sec-desc'>— filter by dimension values to explore raw rows</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    dim_cols_all = rows_sel + [c for c in cols_sel if c not in rows_sel]
    _pv_state = st.session_state.get("data_explorer_pivot") or {}
    _cell_click = _pv_state.get("cell_click") if isinstance(_pv_state, dict) else None
    if isinstance(_cell_click, dict):
        _click_filters: dict[str, str] = _cell_click.get("filters") or {}
        # Guard: only sync when filters are non-empty. The component fires a
        # second cell_click with filters={} after re-rendering; that empty
        # payload must not clear the drill-down values we just set.
        if _click_filters:
            _click_sig = str(sorted(_click_filters.items()))
            if st.session_state.get("_pv_click_sig") != _click_sig:
                st.session_state["_pv_click_sig"] = _click_sig
                for _dim in dim_cols_all:
                    _val = _click_filters.get(_dim)
                    st.session_state[f"_pv_drill_{_dim}"] = [_val] if _val is not None else []

    if dim_cols_all:
        _drill_fragment(df, dim_cols_all)
    else:
        st.caption("Add at least one Rows or Columns field above to enable drill-down.")

