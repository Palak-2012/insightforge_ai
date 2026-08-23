"""
InsightForge AI — Agents Module
===============================
Specialized AI and analytical agents orchestrated via LangGraph.
"""

from insightforge.agents.schema import schema_agent
from insightforge.agents.cleaner import cleaning_agent
from insightforge.agents.eda import eda_agent
from insightforge.agents.viz import viz_agent
from insightforge.agents.insights import insight_agent
from insightforge.agents.reporter import report_agent
from insightforge.agents.supervisor import build_supervisor, run_pipeline

__all__ = [
    "schema_agent",
    "cleaning_agent",
    "eda_agent",
    "viz_agent",
    "insight_agent",
    "report_agent",
    "build_supervisor",
    "run_pipeline",
]
