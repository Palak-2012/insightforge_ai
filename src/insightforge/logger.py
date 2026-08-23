"""
InsightForge AI — Logger Setup
==============================
Structured logging for the multi-agent pipeline.
"""

import datetime
import logging
from typing import Any, List


def setup_logger(name: str = "InsightForge", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def log_event(state: Any, agent: str, message: str) -> List[str]:
    """
    Appends a timestamped log entry to the state's pipeline log and prints it.

    Parameters
    ----------
    state   : InsightForgeState dictionary
    agent   : Name of the calling agent
    message : Log message content

    Returns
    -------
    Updated pipeline_log list
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{agent.upper()}] {message}"
    logger.info(f"[{agent.upper()}] {message}")

    current_log = state.get("pipeline_log", []) if isinstance(state, dict) else []
    return current_log + [entry]
