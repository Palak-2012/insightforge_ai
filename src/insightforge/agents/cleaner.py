"""
InsightForge AI — Agent 2: Cleaning Agent
=========================================
Handles missing value imputation, duplicate removal, and outlier detection.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd

from insightforge.state import InsightForgeState
from insightforge.logger import log_event


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a DataFrame:
    1. Removes exact duplicate rows.
    2. Imputes numeric columns with median.
    3. Imputes categorical columns with mode.
    4. Detects IQR outliers.
    """
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)

    # 1. Deduplication
    duplicates_count = int(cleaned_df.duplicated().sum())
    if duplicates_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # 2. Imputation
    null_counts_before = df.isnull().sum().to_dict()
    imputation_actions = {}
    nulls_filled_total = 0

    for col in cleaned_df.columns:
        null_count = int(cleaned_df[col].isnull().sum())
        if null_count > 0:
            nulls_filled_total += null_count
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                imputation_actions[col] = {"strategy": "median", "value": float(fill_val), "count": null_count}
            else:
                mode_series = cleaned_df[col].mode()
                fill_val = mode_series[0] if not mode_series.empty else "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                imputation_actions[col] = {"strategy": "mode", "value": str(fill_val), "count": null_count}

    # 3. IQR Outlier Detection
    outliers_detected = {}
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            outliers_detected[col] = {
                "count": outlier_count,
                "percentage": round((outlier_count / len(cleaned_df)) * 100, 2),
                "lower_bound": round(float(lower_bound), 3),
                "upper_bound": round(float(upper_bound), 3)
            }

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned_df),
        "duplicates_removed": duplicates_count,
        "nulls_filled_total": nulls_filled_total,
        "null_counts_before": null_counts_before,
        "imputation_actions": imputation_actions,
        "outliers_detected": outliers_detected,
        "total_outliers": sum(item["count"] for item in outliers_detected.values())
    }
    return cleaned_df, report


def cleaning_agent(state: InsightForgeState) -> Dict[str, Any]:
    """
    Agent 2 — Cleaning Agent
    ------------------------
    Reads  : state["raw_df"]
    Writes : state["cleaned_df"], state["cleaning_report"], state["pipeline_log"]
    """
    state["pipeline_log"] = log_event(state, "cleaning_agent", "Starting automated data cleaning...")
    df = state.get("raw_df")
    if df is None:
        state["errors"].append("Cleaning Agent: raw_df is None.")
        return state

    try:
        cleaned_df, report = clean_dataframe(df)
        state["cleaned_df"] = cleaned_df
        state["cleaning_report"] = report
        state["pipeline_log"] = log_event(
            state,
            "cleaning_agent",
            f"Completed: {report['duplicates_removed']} dupes dropped, {report['nulls_filled_total']} nulls filled."
        )
    except Exception as e:
        state["errors"].append(f"Cleaning Agent Error: {str(e)}")
        state["cleaned_df"] = df.copy()
        state["cleaning_report"] = {"error": str(e)}

    return state
