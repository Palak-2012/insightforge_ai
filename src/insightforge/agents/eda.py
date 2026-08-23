"""
InsightForge AI — Agent 3: EDA Agent
====================================
Computes statistical profiles, correlation matrices, and categorical distributions.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd

from insightforge.state import InsightForgeState
from insightforge.logger import log_event


def get_basic_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns shape, memory usage, and column dtypes."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }


def get_statistical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates numerical statistical distribution."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return {}

    summary = {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue
        summary[col] = {
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "q25": round(float(s.quantile(0.25)), 3),
            "q75": round(float(s.quantile(0.75)), 3),
            "skewness": round(float(s.skew()), 3) if len(s) > 2 else 0.0
        }
    return summary


def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes pairwise Pearson correlation for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {}
    corr = num_df.corr().round(3)
    return corr.to_dict()


def get_value_counts(df: pd.DataFrame, max_cols: int = 6, top_n: int = 5) -> Dict[str, Any]:
    """Calculates value distributions for categorical columns."""
    cat_cols = df.select_dtypes(exclude=[np.number, "datetime", "datetimetz"]).columns[:max_cols]
    counts = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(top_n)
        counts[col] = {str(k): int(v) for k, v in vc.items()}
    return counts


def eda_agent(state: InsightForgeState) -> Dict[str, Any]:
    """
    Agent 3 — EDA Agent
    -------------------
    Reads  : state["cleaned_df"] (or raw_df)
    Writes : state["eda_results"], state["pipeline_log"]
    """
    state["pipeline_log"] = log_event(state, "eda_agent", "Running statistical EDA...")
    df = state.get("cleaned_df") if state.get("cleaned_df") is not None else state.get("raw_df")
    if df is None:
        state["errors"].append("EDA Agent: DataFrame is None.")
        return state

    try:
        eda_results = {
            "basic_summary": get_basic_summary(df),
            "statistical_summary": get_statistical_summary(df),
            "correlation_matrix": get_correlation_matrix(df),
            "value_counts": get_value_counts(df)
        }
        state["eda_results"] = eda_results
        num_stats = len(eda_results["statistical_summary"])
        cat_stats = len(eda_results["value_counts"])
        state["pipeline_log"] = log_event(
            state,
            "eda_agent",
            f"Completed: Profiled {num_stats} numeric & {cat_stats} categorical features."
        )
    except Exception as e:
        state["errors"].append(f"EDA Agent Error: {str(e)}")
        state["eda_results"] = {"error": str(e)}

    return state
