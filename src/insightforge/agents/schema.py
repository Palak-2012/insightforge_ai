"""
InsightForge AI — Agent 1: Schema Agent
=======================================
Detects column types, data domain, target variable, and problem type using Gemini AI.
"""

import json
from typing import Any, Dict
import pandas as pd

from insightforge.state import InsightForgeState
from insightforge.logger import log_event


def get_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """Classifies DataFrame columns into numeric, categorical, and datetime."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def schema_agent(state: InsightForgeState) -> Dict[str, Any]:
    """
    Agent 1 — Schema Agent
    ----------------------
    Reads  : state["raw_df"], state["gemini_key"]
    Writes : state["schema_info"], state["pipeline_log"]
    """
    state["pipeline_log"] = log_event(state, "schema_agent", "Starting schema detection...")
    df = state.get("raw_df")
    if df is None:
        state["errors"].append("Schema Agent: raw_df is None.")
        return state

    col_types = get_column_types(df)
    schema_info = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "column_types": col_types,
        "numeric_count": len(col_types["numeric"]),
        "categorical_count": len(col_types["categorical"]),
        "datetime_count": len(col_types["datetime"]),
        "domain": "General Business Data",
        "target_variable": None,
        "problem_type": "Exploratory Analysis"
    }

    gemini_key = state.get("gemini_key", "")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
            Analyze the following dataset columns and sample data to identify:
            1. Business Domain (e.g. Healthcare, Finance, E-Commerce, Transportation, HR)
            2. Likely Target Variable (if any prediction task applies, otherwise 'None')
            3. Problem Type (Classification, Regression, or Descriptive Analysis)

            Columns: {df.columns.tolist()}
            Data Types: {df.dtypes.to_dict()}
            Sample (3 rows):
            {df.head(3).to_dict(orient='records')}

            Return ONLY a valid JSON object matching:
            {{
                "domain": "...",
                "target_variable": "...",
                "problem_type": "...",
                "reasoning": "..."
            }}
            """
            response = model.generate_content(prompt)
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            ai_meta = json.loads(clean_text.strip())
            schema_info.update({
                "domain": ai_meta.get("domain", schema_info["domain"]),
                "target_variable": ai_meta.get("target_variable"),
                "problem_type": ai_meta.get("problem_type", schema_info["problem_type"]),
                "domain_reasoning": ai_meta.get("reasoning", "")
            })
            state["pipeline_log"] = log_event(state, "schema_agent", f"AI detected domain: {schema_info['domain']}")
        except Exception as e:
            state["errors"].append(f"Schema Agent AI detection warning: {str(e)}")
            state["pipeline_log"] = log_event(state, "schema_agent", f"Rule-based schema applied (AI fallback: {e})")

    state["schema_info"] = schema_info
    state["pipeline_log"] = log_event(state, "schema_agent", f"Completed: {schema_info['total_columns']} cols detected.")
    return state
