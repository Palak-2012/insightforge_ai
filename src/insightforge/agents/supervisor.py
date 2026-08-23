"""
InsightForge AI — Supervisor Agent & Pipeline Runner
====================================================
Orchestrates the 7-agent pipeline using LangGraph StateGraph (with sequential fallback).
"""

from typing import Any, Dict, Optional
import pandas as pd

from insightforge.state import InsightForgeState, create_initial_state
from insightforge.logger import log_event
from insightforge.loader import load_dataset
from insightforge.agents.schema import schema_agent
from insightforge.agents.cleaner import cleaning_agent
from insightforge.agents.eda import eda_agent
from insightforge.agents.viz import viz_agent
from insightforge.agents.insights import insight_agent
from insightforge.agents.reporter import report_agent


def build_supervisor():
    """
    Constructs and compiles the LangGraph StateGraph multi-agent pipeline.
    """
    try:
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(InsightForgeState)

        # Register Agent Nodes
        workflow.add_node("schema_agent", schema_agent)
        workflow.add_node("cleaning_agent", cleaning_agent)
        workflow.add_node("eda_agent", eda_agent)
        workflow.add_node("viz_agent", viz_agent)
        workflow.add_node("insight_agent", insight_agent)
        workflow.add_node("report_agent", report_agent)

        # Define Linear Flow
        workflow.set_entry_point("schema_agent")
        workflow.add_edge("schema_agent", "cleaning_agent")
        workflow.add_edge("cleaning_agent", "eda_agent")
        workflow.add_edge("eda_agent", "viz_agent")
        workflow.add_edge("viz_agent", "insight_agent")
        workflow.add_edge("insight_agent", "report_agent")
        workflow.add_edge("report_agent", END)

        return workflow.compile()
    except ImportError:
        # Graceful fallback runner if langgraph is not installed in local environment
        class SequentialSupervisor:
            def invoke(self, state: InsightForgeState) -> InsightForgeState:
                state = schema_agent(state)
                state = cleaning_agent(state)
                state = eda_agent(state)
                state = viz_agent(state)
                state = insight_agent(state)
                state = report_agent(state)
                return state

        return SequentialSupervisor()


def run_pipeline(
    data_source: Any,
    gemini_key: str = "",
    output_pdf: str = "insightforge_report.pdf"
) -> Dict[str, Any]:
    """
    High-level entrypoint to execute the full InsightForge AI multi-agent workflow.

    Parameters
    ----------
    data_source : str (filepath) or pd.DataFrame
    gemini_key  : Optional Gemini API Key
    output_pdf  : Path to save the generated PDF report

    Returns
    -------
    Final pipeline state dictionary
    """
    if isinstance(data_source, str):
        df, info = load_dataset(data_source)
        dataset_path = data_source
    elif isinstance(data_source, pd.DataFrame):
        df = data_source
        dataset_path = "in_memory_dataframe"
    else:
        raise ValueError("data_source must be a filepath string or a pandas DataFrame.")

    initial_state = create_initial_state(
        dataset_path=dataset_path,
        raw_df=df,
        gemini_key=gemini_key
    )
    initial_state["pdf_path"] = output_pdf
    initial_state["pipeline_log"] = log_event(initial_state, "supervisor", "Initialized InsightForge AI Pipeline.")

    supervisor = build_supervisor()
    final_state = supervisor.invoke(initial_state)
    final_state["pipeline_log"] = log_event(final_state, "supervisor", "InsightForge AI Pipeline Completed Successfully!")
    return final_state
