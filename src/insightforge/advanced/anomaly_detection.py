"""
InsightForge AI — Advanced Feature 2: Anomaly & Outlier Detection
================================================================
IQR-based outlier identification, boxplot figures, and Gemini-driven explanations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px

from insightforge.llm import call_gemini


def compute_iqr_bounds(series: pd.Series, factor: float = 1.5) -> Dict[str, float]:
    """Computes IQR lower and upper threshold bounds."""
    s = series.dropna()
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    return {
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_bound": round(q1 - factor * iqr, 3),
        "upper_bound": round(q3 + factor * iqr, 3)
    }


def detect_anomalies(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, Any]:
    """Scans all numerical columns and flags records exceeding IQR limits."""
    results = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        bounds = compute_iqr_bounds(df[col], factor=factor)
        outlier_mask = (df[col] < bounds["lower_bound"]) | (df[col] > bounds["upper_bound"])
        outlier_indices = df[outlier_mask].index.tolist()

        if outlier_indices:
            results[col] = {
                "bounds": bounds,
                "outlier_count": len(outlier_indices),
                "outlier_pct": round((len(outlier_indices) / len(df)) * 100, 2),
                "outlier_indices": outlier_indices[:50],  # cap for readability
                "min_outlier": float(df.loc[outlier_indices, col].min()),
                "max_outlier": float(df.loc[outlier_indices, col].max())
            }

    return results


def explain_anomalies(anomalies: Dict[str, Any], gemini_key: str = "") -> str:
    """Generates an AI-written explanation for why detected outliers are notable."""
    if not gemini_key or not anomalies:
        return "No anomalies detected or Gemini key not provided."

    try:
        prompt = f"""
        Explain the potential business and data implications of these statistical outliers detected via IQR:
        {anomalies}

        Provide 3 bullet points on possible causes and recommended validation actions.
        """
        return call_gemini(prompt, gemini_key)
    except Exception as e:
        return f"Statistical outlier detection completed ({len(anomalies)} columns with bounds exceeding 1.5x IQR)."
