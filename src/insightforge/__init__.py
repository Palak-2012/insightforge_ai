"""
InsightForge AI
===============
A Multi-Agent AI Framework for Automated Data Analysis and Business Insight Generation.
"""

from insightforge.state import InsightForgeState
from insightforge.loader import load_dataset, validate_file
from insightforge.agents.supervisor import build_supervisor, run_pipeline

__version__ = "1.0.0"
__author__ = "Palak Parihar"

__all__ = [
    "InsightForgeState",
    "load_dataset",
    "validate_file",
    "build_supervisor",
    "run_pipeline",
]
