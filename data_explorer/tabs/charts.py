"""Charts tab: histogram, bar, scatter, box plots."""
from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st


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
