"""
InsightForge AI — Advanced Feature 1: Data Dictionary Generator
==============================================================
Generates column-level statistical profiling and semantic definitions using Gemini AI.
"""

from typing import Any, Dict, Optional
import pandas as pd


def get_column_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Extracts profiling statistics for a single column."""
    s = df[col]
    null_count = int(s.isnull().sum())
    unique_count = int(s.nunique())
    sample_vals = [str(v) for v in s.dropna().unique()[:4]]

    return {
        "column": col,
        "dtype": str(s.dtype),
        "null_count": null_count,
        "null_pct": round((null_count / len(df)) * 100, 2),
        "unique_count": unique_count,
        "sample_values": sample_vals
    }


def generate_data_dictionary(df: pd.DataFrame, gemini_key: str = "") -> pd.DataFrame:
    """
    Generates a structured data dictionary DataFrame describing every column.
    """
    rows = []
    model = None
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
        except ImportError:
            model = None

    for col in df.columns:
        stats = get_column_stats(df, col)
        description = f"Attribute '{col}' with {stats['unique_count']} unique values."

        if model:
            try:
                prompt = f"In one concise sentence, explain the business meaning of a column named '{col}' in a dataset where sample values are: {stats['sample_values']}."
                resp = model.generate_content(prompt)
                description = resp.text.strip()
            except Exception:
                pass

        rows.append({
            "Column Name": col,
            "Data Type": stats["dtype"],
            "Unique Count": stats["unique_count"],
            "Null Percentage (%)": stats["null_pct"],
            "Sample Values": ", ".join(stats["sample_values"]),
            "Business Definition": description
        })

    return pd.DataFrame(rows)
