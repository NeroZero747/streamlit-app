"""Cached schema/preview helpers and per-column filter metadata."""
from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import streamlit as st

# Threshold above which a categorical filter switches to a "contains" text input
DROPDOWN_MAX_UNIQUE = 30


@st.cache_data(show_spinner=False)
def schema_summary(df: pl.DataFrame) -> pd.DataFrame:
    rows = []
    for col, dtype in df.schema.items():
        s = df.get_column(col)
        rows.append(
            {
                "column": col,
                "dtype": str(dtype),
                "non_null": int(s.len() - s.null_count()),
                "nulls": int(s.null_count()),
                "unique": int(s.n_unique()),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=4)
def head_pandas(df: pl.DataFrame, n: int) -> pd.DataFrame:
    """Cheap, cached conversion of the head(n) slice for display."""
    return df.head(n).to_pandas()


@st.cache_data(show_spinner=False, max_entries=64)
def column_filter_meta(_df: pl.DataFrame, col: str, df_id: str) -> dict:
    """Compute filter metadata for a single column over the FULL frame (cached).

    `_df` is prefixed with underscore so Streamlit skips hashing the (potentially
    huge) Polars frame. `df_id` is a cheap, stable cache key supplied by the
    caller — change it whenever the underlying data changes.
    """
    s = _df.get_column(col)
    dtype = s.dtype
    meta: dict[str, Any] = {"dtype": str(dtype), "n_unique": int(s.n_unique())}
    if dtype.is_numeric():
        non_null = s.drop_nulls()
        if non_null.len() > 0:
            meta["min"] = non_null.min()
            meta["max"] = non_null.max()
        meta["kind"] = "numeric"
    elif dtype in (pl.Date, pl.Datetime):
        non_null = s.drop_nulls()
        if non_null.len() > 0:
            meta["min"] = non_null.min()
            meta["max"] = non_null.max()
        meta["kind"] = "datetime"
    else:
        # Try to detect date-like string columns and treat them as datetime
        parsed = None
        if dtype == pl.Utf8:
            sample = s.drop_nulls().head(50)
            if sample.len() > 0:
                for fmt in (
                    None,
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d",
                    "%m-%d-%Y",
                ):
                    try:
                        if fmt is None:
                            test = sample.str.to_date(strict=False)
                        else:
                            test = sample.str.to_date(format=fmt, strict=False)
                        if test.null_count() == 0:
                            if fmt is None:
                                parsed = s.str.to_date(strict=False)
                            else:
                                parsed = s.str.to_date(format=fmt, strict=False)
                            break
                    except Exception:
                        continue
        if parsed is not None and parsed.drop_nulls().len() > 0:
            meta["kind"] = "datetime"
            meta["parsed_as_date"] = True
            non_null = parsed.drop_nulls()
            meta["min"] = non_null.min()
            meta["max"] = non_null.max()
        else:
            meta["kind"] = "categorical"
            if meta["n_unique"] <= DROPDOWN_MAX_UNIQUE:
                uniques = s.drop_nulls().unique().sort().to_list()
                meta["uniques"] = [str(v) for v in uniques]
    return meta
