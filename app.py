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
import base64
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

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
      /* ---- Fonts ---- */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

      /* Hide ONLY the Deploy button — keep the rest of the toolbar */
      [data-testid="stAppDeployButton"],
      .stAppDeployButton,
      [data-testid="stToolbar"] button[kind="header"][title*="Deploy" i],
      [data-testid="stToolbar"] a[href*="deploy" i] {
        display: none !important;
      }

      :root {
        /* Brand */
        --accent:#cb187d;
        --accent-hover:#e0249a;
        --accent-soft: rgba(203,24,125,0.10);
        --accent-glow: rgba(203,24,125,0.22);
        --accent2:#e879f9;
        --accent3:#a855f7;

        /* Neutral palette (Linear-inspired) */
        --bg:#0a0d18;
        --bg-elev:#0f1320;
        --surface:#13182a;
        --surface-2:#171d33;
        --surface-hover:#1c2238;

        /* Text */
        --text:#e4e7ee;
        --text-strong:#f8fafc;
        --muted:#8a93a6;
        --muted-2:#5d6478;

        /* Borders & shadows */
        --border:rgba(255,255,255,0.06);
        --border-strong:rgba(255,255,255,0.10);
        --border-accent: rgba(203,24,125,0.40);

        --shadow-sm: 0 1px 2px rgba(0,0,0,0.30);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.30), 0 1px 2px rgba(0,0,0,0.20);
        --shadow-lg: 0 12px 32px rgba(0,0,0,0.40);

        --radius-xs: 6px;
        --radius-sm: 8px;
        --radius-md: 10px;
        --radius-lg: 14px;
      }

      html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        font-feature-settings: "cv11", "ss01", "ss03";
      }
      /* Apply Inter to text elements only — not button/span (breaks icon ligatures) */
      .stApp p,
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
      .stApp label,
      .stApp input,
      .stApp textarea,
      .stApp select,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stText"],
      .stApp [data-testid="stCaption"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      /* Explicitly restore Material Symbols on every button descendant
         (the collapse arrow, tab icons, expander chevrons etc.) */
      button span,
      button svg,
      [data-testid="stSidebarCollapsedControl"] *,
      [data-testid="stSidebar"] button * {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
      }
      /* Styled Streamlit buttons get Inter for their text */
      .stButton > button,
      .stDownloadButton > button,
      [data-testid="stBaseButton-secondary"],
      [data-testid="stBaseButton-primary"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
      }
      code, pre, kbd, samp { font-family: 'JetBrains Mono', ui-monospace, monospace; }

      /* Single calm background wash — no more competing radial gradients */
      .stApp {
        background:
          radial-gradient(1200px 600px at 100% -10%, rgba(203,24,125,0.06), transparent 60%),
          var(--bg);
      }
      .block-container {
        padding-top: 3.2rem;
        padding-bottom: 2.5rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
        max-width: 1600px !important;
      }
      h1, h2, h3, h4 {
        letter-spacing: -0.02em;
        color: var(--text-strong);
        font-weight: 700;
      }
      p, li { color: var(--text); }
      ::selection { background: var(--accent); color: white; }

      header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 999 !important;
      }
      #MainMenu, footer { visibility: hidden; }

      /* Sidebar collapse/expand button — always visible */
      [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
      }
      /* ===== File Uploader — styled native dropzone ===== */
      [data-testid="stFileUploader"] > label {
        display: none !important;
      }
      [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed rgba(203,24,125,0.30) !important;
        border-radius: var(--radius-md) !important;
        background:
          radial-gradient(ellipse 240px 120px at 50% 0%, rgba(203,24,125,0.06), transparent 70%),
          var(--bg-elev) !important;
        padding: 22px 16px 20px !important;
        text-align: center !important;
        transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease !important;
        cursor: pointer !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0 !important;
      }
      [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
        background:
          radial-gradient(ellipse 240px 120px at 50% 0%, rgba(203,24,125,0.12), transparent 70%),
          var(--bg-elev) !important;
      }

      /* Stack instructions vertically, centered */
      [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        color: var(--text) !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0 !important;
        text-align: center !important;
        width: 100% !important;
      }

      /* Icon as a circular accent badge */
      [data-testid="stFileUploaderDropzoneInstructions"] svg {
        color: var(--accent) !important;
        width: 22px !important;
        height: 22px !important;
        opacity: 1 !important;
        display: block !important;
        margin: 0 !important;
        padding: 10px !important;
        box-sizing: content-box !important;
        background: rgba(203,24,125,0.10) !important;
        border: 1px solid rgba(203,24,125,0.25) !important;
        border-radius: 50% !important;
      }

      /* Hide Streamlit's default primary text */
      [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-of-type {
        display: none !important;
      }
      /* Custom title */
      [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
        content: "Drop a file or click to browse";
        display: block;
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--text-strong);
        margin: 12px 0 4px;
        font-family: 'Inter', -apple-system, sans-serif;
        letter-spacing: -0.005em;
      }
      [data-testid="stFileUploaderDropzoneInstructions"] small,
      [data-testid="stFileUploaderDropzoneInstructions"] span small {
        font-size: 0.68rem !important;
        color: var(--muted-2) !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.5 !important;
      }

      /* Browse files CTA */
      [data-testid="stFileUploaderDropzone"] button {
        margin-top: 14px !important;
        background: var(--accent) !important;
        border: none !important;
        color: white !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.42rem 1.15rem !important;
        box-shadow: 0 4px 12px rgba(203,24,125,0.35) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
      }
      [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--accent-hover) !important;
        box-shadow: 0 4px 16px rgba(203,24,125,0.50) !important;
        transform: translateY(-1px);
      }
      /* Hide duplicate icon in browse button only */
      [data-testid="stFileUploaderDropzone"] button:not([data-testid="stFileUploaderDeleteBtn"]) svg,
      [data-testid="stFileUploaderDropzone"] button:not([data-testid="stFileUploaderDeleteBtn"]) [data-testid="stIconMaterial"] {
        display: none !important;
      }

      /* Uploaded file chip */
      [data-testid="stFileUploaderFile"] {
        background: var(--surface) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius-sm) !important;
        margin-top: 10px !important;
        padding: 10px 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
      }
      [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: var(--text-strong) !important;
        font-family: 'Inter', sans-serif !important;
      }
      [data-testid="stFileUploaderFile"] small {
        color: var(--muted-2) !important;
        font-size: 0.68rem !important;
      }
      /* File icon to the left of name */
      [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileData"] svg,
      [data-testid="stFileUploaderFile"] > svg {
        color: var(--accent) !important;
        opacity: 0.85;
      }
      /* Delete (×) button on file chip */
      [data-testid="stFileUploaderDeleteBtn"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--muted) !important;
        border-radius: var(--radius-xs) !important;
        padding: 4px !important;
        margin: 0 !important;
        box-shadow: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
      }
      [data-testid="stFileUploaderDeleteBtn"]:hover {
        background: rgba(239,68,68,0.10) !important;
        border-color: rgba(239,68,68,0.30) !important;
        color: #ef4444 !important;
      }
      [data-testid="stFileUploaderDeleteBtn"] svg,
      [data-testid="stFileUploaderDeleteBtn"] [data-testid="stIconMaterial"] {
        display: inline-block !important;
        width: 18px !important;
        height: 18px !important;
        font-size: 18px !important;
      }

      /* ===== Sidebar ===== */
      [data-testid="stSidebar"] {
        background: #080b17;
        border-right: 1px solid var(--border);
      }
      [data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0;
      }
      [data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
        border-color: var(--border);
      }
      [data-testid="stSidebar"] label {
        color: var(--muted) !important;
        font-weight: 500;
        font-size: 0.78rem;
        text-transform: none;
        letter-spacing: 0;
      }

      /* ---- Sidebar brand header ---- */
      .sb-header {
        padding: 20px 20px 16px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 4px;
      }
      .sb-brand {
        display: flex;
        align-items: center;
        gap: 11px;
      }
      .sb-brand-logo {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, #cb187d 0%, #e879f9 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        color: white;
        box-shadow: 0 4px 12px rgba(203,24,125,0.35);
        flex-shrink: 0;
        line-height: 1;
      }
      .sb-brand-text { line-height: 1.25; }
      .sb-brand-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text-strong);
        letter-spacing: -0.01em;
        display: block;
      }
      .sb-brand-sub {
        color: var(--muted-2);
        font-size: 0.65rem;
        letter-spacing: 0.06em;
        display: block;
        margin-top: 2px;
        text-transform: uppercase;
      }

      /* ---- Sidebar nav body ---- */
      .sb-body {
        padding: 12px 16px;
      }

      /* Section group */
      .sb-group {
        margin-bottom: 20px;
      }
      .sb-group-label {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted-2);
        margin: 0 0 10px 2px;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .sb-group-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border);
      }

      /* Inline badge (e.g. row count) */
      .sb-badge {
        font-size: 0.62rem;
        color: var(--accent);
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 999px;
        background: var(--accent-soft);
        border: 1px solid var(--border-accent);
        letter-spacing: 0;
      }

      /* Section header inside sidebar */
      .sb-section {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0 0 8px 0;
      }
      .sb-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted-2);
      }
      .sb-section-badge {
        font-size: 0.65rem;
        color: var(--muted);
        font-weight: 500;
        padding: 2px 7px;
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
      }

      /* Footer */
      .sb-footer {
        padding: 12px 16px;
        border-top: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.68rem;
        color: var(--muted-2);
      }
      .sb-footer .sb-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }
      .sb-footer .sb-status .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 6px rgba(34,197,94,0.55);
      }
      .sb-footer .sb-version {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
      }

      /* Tighter rhythm for sidebar widgets */
      [data-testid="stSidebar"] [data-testid="stTextInput"],
      [data-testid="stSidebar"] [data-testid="stSelectbox"],
      [data-testid="stSidebar"] [data-testid="stFileUploader"],
      [data-testid="stSidebar"] [data-testid="stToggle"],
      [data-testid="stSidebar"] [data-testid="stSlider"] {
        margin-bottom: 0.5rem;
      }

      /* Ghost button — subtle inline action (used for Clear filters) */
      .stButton > button.ghost,
      button[kind="secondary"].ghost {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0.7rem !important;
        box-shadow: none !important;
      }
      .stButton > button.ghost:hover {
        color: var(--text-strong) !important;
        border-color: var(--border-accent) !important;
        background: var(--accent-soft) !important;
      }

      /* ---- Hero ---- */
      .hero {
        padding: 28px 32px 24px;
        margin-bottom: 24px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        position: relative;
        overflow: hidden;
      }
      .hero::before {
        content: ""; position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--accent) 40%, var(--accent2) 100%);
        opacity: 0.6;
      }
      .hero::after {
        content: ""; position: absolute;
        top: 0; right: 0; width: 340px; height: 100%;
        background: radial-gradient(400px 300px at 100% 0%, rgba(203,24,125,0.08), transparent 70%);
        pointer-events: none;
      }
      .hero-row {
        display: flex; align-items: flex-start; justify-content: space-between;
        gap: 24px; flex-wrap: wrap;
        position: relative; z-index: 1;
      }
      .hero-eyebrow {
        font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.16em;
        color: var(--accent);
        margin-bottom: 12px;
        display: inline-flex; align-items: center; gap: 8px;
      }
      .hero-eyebrow::before {
        content: ""; width: 14px; height: 1px; background: var(--accent);
      }
      .hero-title {
        font-size: 1.85rem; font-weight: 700; line-height: 1.1;
        color: var(--text-strong);
        letter-spacing: -0.03em;
      }
      .hero-sub {
        color: var(--muted); font-size: 0.88rem;
        margin-top: 8px; font-weight: 400;
        max-width: 480px; line-height: 1.6;
      }
      .hero-source {
        margin-top: 16px;
        display: inline-flex; align-items: center; gap: 8px;
        padding: 5px 12px;
        background: var(--bg-elev);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        font-size: 0.78rem;
        color: var(--text);
        font-family: 'JetBrains Mono', monospace !important;
      }
      .hero-source .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.55);
        flex-shrink: 0;
      }

      /* Metric tiles */
      .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 10px;
        margin-top: 20px;
        position: relative; z-index: 1;
      }
      .metric {
        background: var(--bg-elev);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        transition: border-color 0.15s ease;
      }
      .metric:hover { border-color: var(--border-strong); }
      .metric-label {
        font-size: 0.62rem; color: var(--muted-2);
        text-transform: uppercase; letter-spacing: 0.14em;
        font-weight: 700;
      }
      .metric-value {
        font-size: 1.55rem; font-weight: 700;
        color: var(--text-strong);
        margin-top: 6px; line-height: 1;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.025em;
      }
      .metric-value .unit {
        font-size: 0.72rem; color: var(--muted);
        font-weight: 500; margin-left: 3px;
      }

      /* ---- Pills / tags ---- */
      .pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 999px;
        background: rgba(255,255,255,0.04);
        color: var(--text);
        font-size: 0.7rem; font-weight: 500;
        margin-right: 6px;
        border: 1px solid var(--border-strong);
      }
      .pill.cyan { color: #fae8ff; border-color: rgba(232,121,249,0.30); }
      .pill.pink { color: #e9d5ff; border-color: rgba(168,85,247,0.30); }

      .glass {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 22px 24px;
        box-shadow: var(--shadow-md);
      }

      /* ---- Buttons ---- */
      .stButton > button, .stDownloadButton > button {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-strong);
        background: var(--surface-2);
        color: var(--text);
        font-weight: 500;
        font-size: 0.88rem;
        transition: background 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
        padding: 0.5rem 1rem;
        box-shadow: var(--shadow-sm);
      }
      .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--border-accent);
        background: var(--surface-hover);
        box-shadow: 0 0 0 3px var(--accent-soft);
      }
      .stButton > button:active, .stDownloadButton > button:active {
        background: var(--surface);
      }

      /* ---- Inputs ---- */
      [data-testid="stTextInput"] input,
      [data-testid="stNumberInput"] input,
      [data-testid="stDateInput"] input,
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
      [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background-color: var(--bg-elev) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-strong) !important;
        transition: border-color 0.12s ease, box-shadow 0.12s ease;
      }
      [data-testid="stTextInput"] input:focus,
      [data-testid="stNumberInput"] input:focus,
      [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
      [data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
      }
      [data-testid="stFileUploader"] section {
        background: var(--bg-elev);
        border: 1.5px dashed var(--border-strong);
        border-radius: var(--radius-md);
        transition: border-color 0.15s ease, background 0.15s ease;
      }
      [data-testid="stFileUploader"] section:hover {
        background: var(--surface);
        border-color: var(--accent);
      }

      /* Sliders */
      [data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
      }
      [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) > div {
        background: var(--accent) !important;
      }
      /* Always-visible thumb value bubble (current value above thumb) */
      [data-testid="stSlider"] [data-testid="stThumbValue"],
      [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] + div,
      [data-testid="stSlider"] [data-baseweb="thumb-value"] {
        opacity: 1 !important;
        visibility: visible !important;
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text-strong) !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
      }
      /* Min / max tick labels under the track — always visible, white */
      [data-testid="stSlider"] [data-testid="stTickBar"],
      [data-testid="stSlider"] [data-testid="stTickBarMin"],
      [data-testid="stSlider"] [data-testid="stTickBarMax"],
      [data-testid="stSlider"] [data-baseweb="slider"] > div:last-child,
      [data-testid="stSlider"] [data-baseweb="slider"] > div:last-child > div {
        color: var(--text-strong) !important;
        font-weight: 500 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        background: transparent !important;
        background-color: transparent !important;
        opacity: 1 !important;
        visibility: visible !important;
      }

      /* Toggles */
      [data-testid="stToggle"] { padding: 4px 0; }
      [data-testid="stToggle"] [role="checkbox"][aria-checked="true"] {
        background: var(--accent) !important;
      }

      /* ---- Data grid ---- */
      div[data-testid="stDataFrame"] {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        background: var(--surface);
      }

      /* ---- Tabs — underline editorial style ---- */
      .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border: none;
        border-bottom: 1px solid var(--border);
        border-radius: 0;
        padding: 0;
        margin-bottom: 18px;
      }
      .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: transparent !important;
        border-radius: 0;
        padding: 0 18px;
        color: var(--muted);
        font-weight: 500;
        font-size: 0.88rem;
        border: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: color 0.12s ease, border-color 0.12s ease;
        letter-spacing: 0;
      }
      .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-strong);
      }
      .stTabs [aria-selected="true"] {
        color: var(--text-strong) !important;
        border-bottom-color: var(--accent) !important;
        font-weight: 600;
        background: transparent !important;
        box-shadow: none !important;
      }
      .stTabs [data-baseweb="tab-highlight"] { display: none; }
      .stTabs [data-baseweb="tab-border"] { display: none; }

      /* ---- Expanders ---- */
      [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface);
        overflow: hidden;
        margin-bottom: 12px;
        box-shadow: var(--shadow-sm);
      }
      [data-testid="stExpander"] summary {
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.85rem 1.1rem;
        transition: background 0.12s ease;
      }
      [data-testid="stExpander"] summary:hover {
        background: var(--surface-2);
      }

      /* ---- Captions ---- */
      .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        font-weight: 400;
      }

      /* ---- Alerts ---- */
      [data-testid="stAlert"] {
        border-radius: var(--radius-md);
        border: 1px solid var(--border-strong);
      }

      /* ---- Radio (filter mode bar) ---- */
      [data-testid="stRadio"] [role="radiogroup"] {
        gap: 4px;
        background: var(--bg-elev);
        border: 1px solid var(--border);
        padding: 4px;
        border-radius: var(--radius-sm);
        display: inline-flex;
      }
      [data-testid="stRadio"] [role="radiogroup"] label {
        padding: 6px 12px;
        border-radius: 6px;
        margin: 0;
        cursor: pointer;
        font-size: 0.82rem;
        transition: background 0.12s ease;
      }
      [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background: var(--surface-hover);
      }

      /* ---- Scrollbar ---- */
      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.10);
        border-radius: 10px;
        border: 2px solid transparent;
        background-clip: content-box;
      }
      ::-webkit-scrollbar-thumb:hover {
        background: rgba(203,24,125,0.45);
        background-clip: content-box;
      }
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


# Threshold above which a categorical filter switches to a "contains" text input
DROPDOWN_MAX_UNIQUE = 30


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
                            # Parse the full column
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
            # Only materialize unique values when small enough for a dropdown
            if meta["n_unique"] <= DROPDOWN_MAX_UNIQUE:
                uniques = s.drop_nulls().unique().sort().to_list()
                meta["uniques"] = [str(v) for v in uniques]
    return meta


def polars_filter_ui(df: pl.DataFrame) -> pl.DataFrame:
    """
    Custom filter UI driven by the FULL Polars frame.

    - Numeric / date columns: range slider over true min/max of the full frame.
    - Categorical with <= DROPDOWN_MAX_UNIQUE unique values: multiselect with
      ALL distinct values from the full frame.
    - Categorical with > DROPDOWN_MAX_UNIQUE unique values: "contains" text input.

    Returns a filtered Polars frame.
    """
    # Stable cache key for this frame instance (cheap; doesn't hash data).
    df_id = f"{id(df)}-{df.height}-{df.width}"
    # Nonce lets us reset every widget below by simply bumping it.
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
            # Only show Clear when there's something to clear — keeps the
            # default state quiet.
            if currently_active or chosen:
                clear_clicked = st.button(
                    "Clear",
                    key=f"clear_filters_btn_{nonce}",
                    use_container_width=True,
                    help="Reset all active filters.",
                    type="secondary",
                )
                # Apply ghost styling to this specific button
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
                    # Scope the rerun to just the grid fragment so the rest of
                    # the app (hero, sidebar, schema/charts tabs) doesn't
                    # re-execute. Falls back to a full rerun on older Streamlit.
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
                # Normalize bounds to date for the widgets
                dmin_raw, dmax_raw = meta["min"], meta["max"]
                dmin = dmin_raw.date() if isinstance(dmin_raw, datetime) else dmin_raw
                dmax = dmax_raw.date() if isinstance(dmax_raw, datetime) else dmax_raw

                # If the source column is a string we parsed as date, build an
                # expression that parses it on the fly for filtering.
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
                    # Clamp to data bounds
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
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

    # --- Display -----------------------------------------------------------
    _badge = ""
    if df is not None:
        _badge = f"<span class='sb-badge'>{df.height:,} rows</span>"
    st.markdown(
        f"""
        <div class="sb-group-label" style="margin-top:20px">Display {_badge}</div>
        """,
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

    # --- Footer ------------------------------------------------------------
    _status = "Ready" if df is not None else "Awaiting data"
    st.markdown(
        f"""
        <div class="sb-footer">
          <div class="sb-status"><span class="dot"></span>{_status}</div>
          <div class="sb-version">v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main — Hero header
# ---------------------------------------------------------------------------
if df is None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-row">
            <div>
              <div class="hero-eyebrow">Workspace</div>
              <div class="hero-title">Data Explorer</div>
              <div class="hero-sub">Upload a file or connect to a Posit Connect pin to start exploring.</div>
              <div style="margin-top:14px">
                <span class='pill'>Polars</span>
                <span class='pill cyan'>Glide Data Grid</span>
                <span class='pill pink'>Plotly</span>
                <span class='pill cyan'>Posit Pins</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='glass' style='margin-top:24px;text-align:center;padding:60px 24px'>
          <div style="font-size:48px;margin-bottom:8px">📂</div>
          <h3 style="margin:0">No data loaded yet</h3>
          <p style='color:#94a3b8;margin-top:6px'>
            Use the sidebar to upload CSV, TSV, Parquet, JSON/NDJSON, or Excel — up to 5 GB.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Stash for fragments
st.session_state["_df"] = df
st.session_state["_source_label"] = source_label
st.session_state["_use_filter_ui"] = use_filter_ui
st.session_state["_max_render"] = max_render

# Hero header with inline stats
import html

_lbl = html.escape(source_label) if source_label else "—"
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-row">
        <div>
          <div class="hero-eyebrow">Workspace</div>
          <div class="hero-title">Data Explorer</div>
          <div class="hero-sub">Inspect, filter, visualize, and export your dataset.</div>
          <div class="hero-source"><span class="dot"></span>{_lbl}</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-self:flex-start">
          <span class='pill'>Polars</span>
          <span class='pill cyan'>Glide</span>
          <span class='pill pink'>Plotly</span>
        </div>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Rows</div>
          <div class="metric-value">{df.height:,}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Columns</div>
          <div class="metric-value">{df.width}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Memory</div>
          <div class="metric-value">{df.estimated_size("mb"):.1f}<span class="unit">MB</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_grid, tab_charts, tab_schema = st.tabs(["  Grid  ", "  Charts  ", "  Schema  "])


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
        # Custom Polars-backed filter UI: dropdown/range options come from the
        # FULL frame (not just the rendered head), so users see every value.
        filtered_pl = polars_filter_ui(work)
        rendered = head_pandas(filtered_pl, min(max_render, filtered_pl.height))
        st.caption(
            f"Previewing first {len(rendered):,} of {filtered_pl.height:,} filtered rows "
            f"(out of {work.height:,} total). Downloads export the full filtered set."
        )
        st.dataframe(
            rendered,
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        # Downloads use the FULL filtered frame, not just the previewed head
        export_pl = filtered_pl
    else:
        rendered = head_pandas(work, min(max_render, work.height))
        st.caption(
            f"Previewing first {len(rendered):,} of {work.height:,} rows. "
            "Use Glide's column header menus for sort/search. "
            "Downloads export the full dataset."
        )
        st.dataframe(
            rendered,
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        export_pl = work

    # Cache built download payloads against a cheap signature of the export
    # frame so we don't re-serialize CSV/Parquet on every filter change.
    export_sig = f"{id(df)}-{export_pl.height}-{export_pl.width}-{hash(tuple(export_pl.columns))}"
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

    # Default base name for the suggested file in the Save As dialog.
    base_name = "data"

    def _save_as_button(
        label: str,
        sublabel: str,
        icon_svg: str,
        accent: str,
        payload_fn,
        suggested_name: str,
        mime: str,
        ext: str,
        accept_desc: str,
        component_key: str,
    ) -> None:
        """Render a styled download button.

        A hidden native st.download_button (with a lazy callable) sits behind
        the visual HTML button. No bytes are built at render time — Streamlit
        only calls payload_fn() when the user actually clicks.
        """
        anchor_id = f"__dl_anchor_{component_key}__"

        # 1. Invisible anchor marker so the iframe JS can locate the native button.
        st.markdown(
            f'<span id="{anchor_id}" style="display:none"></span>',
            unsafe_allow_html=True,
        )
        # 2. Hidden native download button — bytes built lazily on click only.
        st.download_button(
            label="dl",
            data=payload_fn,
            file_name=suggested_name,
            mime=mime,
            key=f"_hb_{component_key}",
            use_container_width=False,
        )

        # 3. CSS to hide the anchor marker row and the native button row.
        st.markdown(
            f"""<style>
  div[data-testid="stMarkdown"]:has(#{anchor_id}),
  div[data-testid="stMarkdown"]:has(#{anchor_id}) + div,
  div[data-testid="stElementContainer"]:has(#{anchor_id}),
  div[data-testid="stElementContainer"]:has(#{anchor_id}) + div {{
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
  }}
</style>""",
            unsafe_allow_html=True,
        )

        # 4. Visual HTML button — identical design, JS clicks the hidden button.
        height = 116
        elem_id = f"sa_{component_key}"
        html_doc = f"""
<!doctype html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ overflow: visible !important; }}
  body {{
    margin: 0; padding: 8px 4px 16px 4px;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: transparent;
  }}
  .save-btn {{
    width: 100%;
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.65rem 0.85rem;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.25);
    background: linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,41,59,0.85));
    color: #e2e8f0;
    font-size: 0.92rem; font-weight: 600;
    cursor: pointer;
    transition: transform 0.12s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    text-align: left;
    backdrop-filter: blur(6px);
  }}
  .save-btn:hover {{
    transform: translateY(-1px);
    border-color: {accent};
    box-shadow: 0 8px 24px -10px {accent}, 0 0 0 1px {accent}33 inset;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95));
  }}
  .save-btn:active {{ transform: translateY(0); }}
  .save-btn:focus-visible {{ outline: 2px solid {accent}; outline-offset: 2px; }}
  .icon {{
    flex: 0 0 36px; height: 36px; width: 36px;
    display: grid; place-items: center;
    border-radius: 9px;
    background: {accent}22;
    color: {accent};
  }}
  .icon svg {{ width: 18px; height: 18px; }}
  .text {{ display: flex; flex-direction: column; line-height: 1.15; min-width: 0; }}
  .text .label {{ font-size: 0.95rem; }}
  .text .sub {{ font-size: 0.72rem; color: #94a3b8; font-weight: 500; margin-top: 2px; }}
  .status {{
    font-size: 0.74rem; color: #94a3b8;
    margin-top: 0.35rem; padding-left: 0.2rem; min-height: 1em;
  }}
  .ok {{ color: #34d399; }}
  .err {{ color: #f87171; }}
</style></head>
<body>
  <button id="{elem_id}-btn" class="save-btn" type="button" aria-label="{label}">
    <span class="icon">{icon_svg}</span>
    <span class="text">
      <span class="label">{label}</span>
      <span class="sub">{sublabel}</span>
    </span>
  </button>
  <div class="status" id="{elem_id}-msg"></div>
<script>
(function() {{
  const anchorId = "{anchor_id}";
  const suggestedName = {suggested_name!r};
  const mime = {mime!r};
  const ext = {ext!r};
  const acceptDesc = {accept_desc!r};
  const btn = document.getElementById("{elem_id}-btn");
  const msg = document.getElementById("{elem_id}-msg");

  // Find the hidden st.download_button container inside the parent document.
  function findDlContainer(retries, cb) {{
    try {{
      const doc = window.parent.document;
      const marker = doc.getElementById(anchorId);
      if (marker) {{
        let scope = marker.closest('[data-testid="stColumn"]');
        if (!scope) {{
          let el = marker.parentElement;
          for (let i = 0; i < 20 && el; i++, el = el.parentElement) {{
            if (el.querySelector('[data-testid="stDownloadButton"]')) {{ scope = el; break; }}
          }}
        }}
        if (scope) {{
          const container = scope.querySelector('[data-testid="stDownloadButton"]');
          if (container) {{ cb(container); return; }}
        }}
      }}
    }} catch(e) {{}}
    if (retries > 0) setTimeout(function() {{ findDlContainer(retries - 1, cb); }}, 150);
    else {{ msg.textContent = "Download failed \u2014 try refreshing."; msg.className = "status err"; }}
  }}

  async function doSaveAs(container) {{
    // Prefer fetching via the anchor href so we can pipe into showSaveFilePicker.
    const a = container.querySelector('a[href]');
    if (a && a.href) {{
      msg.textContent = "Preparing\u2026";
      msg.className = "status ok";
      try {{
        const resp = await fetch(a.href);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const blob = await resp.blob();

        if (window.showSaveFilePicker) {{
          const handle = await window.showSaveFilePicker({{
            suggestedName: suggestedName,
            types: [{{ description: acceptDesc, accept: {{ [mime]: [ext] }} }}],
          }});
          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
          msg.textContent = "\u2713 Saved: " + handle.name;
          msg.className = "status ok";
        }} else {{
          // Fallback for Firefox/Safari: blob URL download
          const url = URL.createObjectURL(blob);
          const dl = document.createElement("a");
          dl.href = url; dl.download = suggestedName;
          document.body.appendChild(dl); dl.click(); dl.remove();
          URL.revokeObjectURL(url);
          msg.textContent = "Downloaded (Save As\u2026 requires Chrome/Edge).";
          msg.className = "status ok";
        }}
      }} catch(e) {{
        if (e && e.name === "AbortError") {{
          msg.textContent = "Cancelled.";
          msg.className = "status";
        }} else {{
          msg.textContent = "Error: " + (e.message || e);
          msg.className = "status err";
        }}
      }}
    }} else {{
      // No anchor href available — fall back to clicking the native button.
      const nativeBtn = container.querySelector('button');
      if (nativeBtn) {{
        nativeBtn.click();
        msg.textContent = "Downloading\u2026";
        msg.className = "status ok";
        setTimeout(function() {{ msg.textContent = ""; }}, 2500);
      }} else {{
        msg.textContent = "Download failed \u2014 try refreshing.";
        msg.className = "status err";
      }}
    }}
  }}

  btn.addEventListener("click", function() {{
    findDlContainer(5, doSaveAs);
  }});
}})();
</script>
</body></html>
"""
        st.components.v1.html(html_doc, height=height)

    # Inline SVG icons (currentColor lets the CSS accent color them)
    icon_csv = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<path d="M8 13h2M8 17h2M14 13h2M14 17h2"/></svg>'
    )
    icon_parquet = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="4" width="18" height="4" rx="1"/>'
        '<rect x="3" y="10" width="18" height="4" rx="1"/>'
        '<rect x="3" y="16" width="18" height="4" rx="1"/></svg>'
    )
    icon_archive = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 8v13H3V8"/><path d="M1 3h22v5H1z"/>'
        '<line x1="10" y1="12" x2="14" y2="12"/></svg>'
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        _save_as_button(
            label="Save as CSV",
            sublabel="Filtered · text/csv",
            icon_svg=icon_csv,
            accent="#cb187d",
            payload_fn=_csv_bytes,
            suggested_name=f"{base_name}.csv",
            mime="text/csv",
            ext=".csv",
            accept_desc="CSV file",
            component_key="csv",
        )
    with d2:
        _save_as_button(
            label="Save as Parquet",
            sublabel="Filtered · columnar",
            icon_svg=icon_parquet,
            accent="#e879f9",
            payload_fn=_parquet_bytes,
            suggested_name=f"{base_name}.parquet",
            mime="application/octet-stream",
            ext=".parquet",
            accept_desc="Parquet file",
            component_key="parquet",
        )
    with d3:
        _save_as_button(
            label="Save full Parquet",
            sublabel="Unfiltered · all rows",
            icon_svg=icon_archive,
            accent="#a855f7",
            payload_fn=_full_parquet_bytes,
            suggested_name=f"{base_name}_full.parquet",
            mime="application/octet-stream",
            ext=".parquet",
            accept_desc="Parquet file",
            component_key="full_parquet",
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
        c for c in cols if c not in numeric_cols and df.get_column(c).n_unique() <= 200
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
    pdf = (
        df.sample(n=sample_n, seed=0).to_pandas()
        if df.height > sample_n
        else df.to_pandas()
    )

    fig = None
    if chart_type == "Histogram" and numeric_cols:
        c = st.selectbox("Column", numeric_cols)
        color = st.selectbox("Color by", ["(none)"] + cat_cols)
        fig = px.histogram(
            pdf,
            x=c,
            color=None if color == "(none)" else color,
            template=template,
            color_discrete_sequence=color_seq,
            nbins=50,
        )
    elif chart_type == "Bar (count)" and cat_cols:
        c = st.selectbox("Column", cat_cols)
        vc = pdf[c].value_counts().reset_index().head(50)
        vc.columns = [c, "count"]
        fig = px.bar(
            vc, x=c, y="count", template=template, color_discrete_sequence=color_seq
        )
    elif chart_type == "Scatter" and len(numeric_cols) >= 2:
        cc1, cc2, cc3 = st.columns(3)
        x = cc1.selectbox("X", numeric_cols, index=0)
        y = cc2.selectbox("Y", numeric_cols, index=1)
        color = cc3.selectbox("Color by", ["(none)"] + cat_cols)
        fig = px.scatter(
            pdf,
            x=x,
            y=y,
            color=None if color == "(none)" else color,
            template=template,
            color_discrete_sequence=color_seq,
            opacity=0.7,
        )
    elif chart_type == "Box" and numeric_cols:
        y = st.selectbox("Y", numeric_cols)
        x = st.selectbox("Group by", ["(none)"] + cat_cols)
        fig = px.box(
            pdf,
            y=y,
            x=None if x == "(none)" else x,
            template=template,
            color_discrete_sequence=color_seq,
        )

    if fig is None:
        st.info("Not enough columns of the right type for this chart.")
        return
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=560,
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
                sample.T,
                aspect="auto",
                color_continuous_scale=["#0b1020", "#cb187d"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10),
                height=480,
                coloraxis_showscale=False,
            )
            fig.update_xaxes(showticklabels=False, title="rows (sample)")
            st.plotly_chart(fig, use_container_width=True, theme=None)


with tab_schema:
    schema_fragment()
