"""
InsightForge AI — Agent 4: Visualization Agent
==============================================
Automatically generates Plotly visualizations matching column types and distributions.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from insightforge.state import InsightForgeState
from insightforge.logger import log_event


def generate_charts(df: pd.DataFrame, max_charts: int = 6) -> List[Any]:
    """Generates standard exploratory Plotly figures (or metadata specs if plotly is unavailable)."""
    charts = []

    try:
        import plotly.express as px

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number, "datetime", "datetimetz"]).columns.tolist()

        # 1. Numeric Histograms
        for col in numeric_cols[:2]:
            fig = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}",
                template="plotly_white",
                marginal="box"
            )
            charts.append(fig)

        # 2. Categorical Bar Charts
        for col in categorical_cols[:2]:
            top_cats = df[col].value_counts().head(8).reset_index()
            top_cats.columns = [col, "Count"]
            fig = px.bar(
                top_cats,
                x=col,
                y="Count",
                title=f"Top Categories in {col}",
                template="plotly_white",
                color="Count",
                color_continuous_scale="Blues"
            )
            charts.append(fig)

        # 3. Correlation Heatmap
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols[:6]].corr().round(2)
            fig = px.imshow(
                corr,
                text_auto=True,
                title="Numerical Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                template="plotly_white"
            )
            charts.append(fig)

        # 4. Scatter Plot (Bivariate)
        if len(numeric_cols) >= 2:
            color_col = categorical_cols[0] if categorical_cols else None
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color=color_col,
                title=f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}",
                template="plotly_white"
            )
            charts.append(fig)

    except ImportError:
        # Fallback chart metadata descriptors
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:2]:
            charts.append({"type": "histogram", "column": col, "title": f"Distribution of {col}"})

    return charts[:max_charts]


def viz_agent(state: InsightForgeState) -> Dict[str, Any]:
    """
    Agent 4 — Visualization Agent
    -----------------------------
    Reads  : state["cleaned_df"] (or raw_df)
    Writes : state["charts"], state["pipeline_log"]
    """
    state["pipeline_log"] = log_event(state, "viz_agent", "Generating automated Plotly visualizations...")
    df = state.get("cleaned_df") if state.get("cleaned_df") is not None else state.get("raw_df")
    if df is None:
        state["errors"].append("Viz Agent: DataFrame is None.")
        return state

    try:
        charts = generate_charts(df)
        state["charts"] = charts
        state["pipeline_log"] = log_event(state, "viz_agent", f"Generated {len(charts)} chart specifications.")
    except Exception as e:
        state["errors"].append(f"Viz Agent Error: {str(e)}")
        state["charts"] = []

    return state
