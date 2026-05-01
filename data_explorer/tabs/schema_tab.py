"""Schema tab: column overview + null heatmap."""
from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from ..schema import schema_summary


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
