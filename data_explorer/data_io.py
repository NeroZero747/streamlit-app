"""Cached data loaders: file uploads and Posit Connect pins."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import streamlit as st


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
