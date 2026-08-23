"""
InsightForge AI — Advanced Features Module
==========================================
Data Dictionary, Outlier / Anomaly Detection, Trend Forecasting, and AutoML.
"""

from insightforge.advanced.data_dictionary import generate_data_dictionary
from insightforge.advanced.anomaly_detection import detect_anomalies, explain_anomalies
from insightforge.advanced.forecasting import generate_trend_forecast
from insightforge.advanced.automl import train_baseline_model

__all__ = [
    "generate_data_dictionary",
    "detect_anomalies",
    "explain_anomalies",
    "generate_trend_forecast",
    "train_baseline_model",
]
