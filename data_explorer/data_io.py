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


@st.cache_data(show_spinner=False, ttl=300)
def get_pin_files(server_url: str, api_key: str, pin_name: str) -> list[str]:
    """Return the list of file names bundled inside a pin."""
    board = get_connect_board(server_url, api_key)
    meta = board.pin_meta(pin_name)
    files = meta.file if isinstance(meta.file, list) else [meta.file]
    return [f for f in files if f]  # drop any None/empty entries


@st.cache_data(show_spinner="Loading file from pin…", max_entries=4)
def load_pin_file(server_url: str, api_key: str, pin_name: str, file_name: str) -> pl.DataFrame:
    """Fetch a specific file from a multi-file pin and load it as a DataFrame."""
    import os
    board = get_connect_board(server_url, api_key)
    fetched = board.pin_fetch(pin_name)  # returns list of local temp paths
    # Find the matching file among fetched paths
    target = next(
        (p for p in fetched if os.path.basename(p) == file_name),
        fetched[0] if fetched else None,
    )
    if target is None:
        raise FileNotFoundError(f"File '{file_name}' not found in pin '{pin_name}'")
    return load_uploaded(file_name, open(target, "rb").read())
