"""
InsightForge AI — Advanced Feature 4: Automated Baseline Machine Learning (AutoML)
=================================================================================
Automates baseline model training, evaluation metrics, and feature importance computation.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def train_baseline_model(
    df: pd.DataFrame,
    target_col: str,
    problem_type: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a baseline Random Forest model for classification or regression.

    Parameters
    ----------
    df           : Cleaned DataFrame
    target_col   : Name of the target variable column
    problem_type : 'Classification', 'Regression', or None (auto-detected)

    Returns
    -------
    Dictionary with model metrics, feature importances, and Plotly figure
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}

    data = df.dropna(subset=[target_col]).copy()
    if len(data) < 10:
        return {"error": "Dataset too small for model training (need at least 10 rows)."}

    # Separate X and y
    y_raw = data[target_col]
    X_raw = data.drop(columns=[target_col])

    # Auto-detect problem type if not supplied
    if not problem_type or problem_type == "Exploratory Analysis":
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
            problem_type = "Regression"
        else:
            problem_type = "Classification"

    # Preprocess features (drop high-cardinality text / IDs)
    cols_to_keep = [c for c in X_raw.columns if X_raw[c].nunique() < len(X_raw) * 0.9]
    X_raw = X_raw[cols_to_keep]

    # Encode categorical features
    X = pd.get_dummies(X_raw, drop_first=True)
    if X.empty:
        return {"error": "No valid predictive features remaining after preprocessing."}

    # Encode target if classification
    if problem_type == "Classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if problem_type == "Classification":
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = round(float(accuracy_score(y_test, y_pred)), 3)
        f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 3)
        metrics = {"Accuracy": acc, "F1_Score": f1}
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
        r2 = round(float(r2_score(y_test, y_pred)), 3)
        metrics = {"RMSE": rmse, "R2_Score": r2}

    # Feature Importance
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top 10 Feature Importances ({problem_type}: {target_col})",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    return {
        "problem_type": problem_type,
        "target_col": target_col,
        "metrics": metrics,
        "top_features": feat_df.to_dict(orient="records"),
        "fig": fig,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
