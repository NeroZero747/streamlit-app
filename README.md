# ⚡ Streamlit Data Explorer

Fast, modern Streamlit app for exploring tabular data with **Polars** and **Pandas**, with optional **Posit Connect pins** integration.

## Features

- 📁 Upload CSV, TSV, Parquet, JSON/NDJSON, Excel
- 📌 Read pins from a Posit Connect server
- 🔎 Per-column live filters (numeric range, categorical, text contains)
- 👁 Toggle column visibility
- ⚡ Polars-powered filtering + `st.fragment` for **isolated reactive reruns** (no full-app delay — closest Streamlit equivalent to Shiny for Python reactivity)
- 🐼 Switch display engine between Polars and Pandas
- ⬇ Download filtered results as CSV or Parquet

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Posit Connect

Either enter the server URL and API key in the sidebar, or set them as environment variables before launching:

```powershell
$env:CONNECT_SERVER = "https://connect.example.com"
$env:CONNECT_API_KEY = "xxxxxxxxxxxxxxxx"
streamlit run app.py
```

The app uses the [`pins`](https://rstudio.github.io/pins-python/) package — pins stored as `pandas.DataFrame`, `polars.DataFrame`, or arrow/parquet objects are auto-converted.

## Why it feels fast

| Technique | What it gives you |
|---|---|
| `@st.fragment` | Filter widget changes rerun **only the table fragment** — not the whole script. This is Streamlit's answer to Shiny's reactive graph. |
| `@st.cache_data` on `load_*` and `apply_filters` | Filter recomputation is memoized on input hashes — re-toggling a value is instant. |
| Polars filtering | Multi-million-row filters in milliseconds; `to_pandas()` only on display. |
| `st.dataframe` (not `st.table`) | Virtualized, GPU-accelerated client-side rendering. |
