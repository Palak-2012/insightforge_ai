"""
InsightForge AI — State Schema
==============================
Typed shared state passed through all agents in the LangGraph pipeline.
"""

from typing import Any, Dict, List, Optional, TypedDict


class InsightForgeState(TypedDict):
    """
    Shared state dictionary managed by LangGraph across the 7 agents.

    Fields
    ------
    dataset_path    : Path to the input file (CSV/Excel)
    gemini_key      : Gemini API key provided at runtime
    raw_df          : Original uploaded DataFrame
    cleaned_df      : DataFrame after cleaning agent processes it
    schema_info     : Detected column types, domain, target variable
    cleaning_report : Summary of missing values filled, dupes removed, outliers
    eda_results     : Descriptive statistics, correlations, distributions
    charts          : List of Plotly Figure objects generated
    insights        : Business insights and recommendations from Gemini AI
    pdf_path        : Path where the generated PDF report is saved
    errors          : Any non-fatal errors or warnings logged during execution
    pipeline_log    : Timestamped list of event logs across agent stages
    """

    dataset_path: str
    gemini_key: str
    raw_df: Any
    cleaned_df: Optional[Any]
    schema_info: Dict[str, Any]
    cleaning_report: Dict[str, Any]
    eda_results: Dict[str, Any]
    charts: List[Any]
    insights: str
    pdf_path: str
    errors: List[str]
    pipeline_log: List[str]


def create_initial_state(
    dataset_path: str = "",
    raw_df: Any = None,
    gemini_key: str = ""
) -> InsightForgeState:
    """Creates a blank, correctly typed initial state for the pipeline."""
    return {
        "dataset_path": dataset_path,
        "gemini_key": gemini_key,
        "raw_df": raw_df,
        "cleaned_df": None,
        "schema_info": {},
        "cleaning_report": {},
        "eda_results": {},
        "charts": [],
        "insights": "",
        "pdf_path": "",
        "errors": [],
        "pipeline_log": [],
    }
