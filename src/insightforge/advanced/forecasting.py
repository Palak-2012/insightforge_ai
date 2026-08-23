"""
InsightForge AI — Advanced Feature 3: Trend & Time-Series Forecasting
=====================================================================
Linear and polynomial trend projection for numerical metrics over time.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def generate_trend_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    periods_ahead: int = 6
) -> Dict[str, Any]:
    """
    Fits a linear trend on date-indexed numerical columns and projects future periods.
    """
    clean_df = df[[date_col, target_col]].dropna().copy()
    clean_df[date_col] = pd.to_datetime(clean_df[date_col])
    clean_df = clean_df.sort_values(date_col)

    if len(clean_df) < 3:
        return {"error": "Need at least 3 historical data points to generate forecast."}

    clean_df["time_idx"] = np.arange(len(clean_df))
    X = clean_df[["time_idx"]].values
    y = clean_df[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    future_idx = np.arange(len(clean_df), len(clean_df) + periods_ahead).reshape(-1, 1)
    future_preds = model.predict(future_idx)

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean_df[date_col],
        y=y,
        mode="lines+markers",
        name="Historical Actuals",
        line=dict(color="#0d6efd")
    ))

    last_date = clean_df[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=periods_ahead + 1, freq="M")[1:]
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode="lines+markers",
        name="Projected Forecast",
        line=dict(dash="dash", color="#dc3545")
    ))

    fig.update_layout(
        title=f"Trend Projection: {target_col} over {date_col}",
        xaxis_title=date_col,
        yaxis_title=target_col,
        template="plotly_white"
    )

    slope = float(model.coef_[0])
    direction = "Upward Growth" if slope > 0 else "Downward Trend" if slope < 0 else "Stable"

    return {
        "fig": fig,
        "trend_direction": direction,
        "slope_per_period": round(slope, 3),
        "r2_score": round(float(model.score(X, y)), 3),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_values": [round(float(v), 2) for v in future_preds]
    }
