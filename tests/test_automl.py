import pandas as pd
from insightforge.advanced.automl import train_baseline_model


def test_train_baseline_classification():
    df = pd.DataFrame({
        "Age": [22, 38, 26, 35, 54, 2, 27, 14, 4, 58, 20, 24, 30, 45, 50],
        "Fare": [7.25, 71.83, 7.92, 53.1, 8.05, 21.07, 11.13, 30.07, 16.7, 26.55, 8.0, 15.0, 25.0, 80.0, 90.0],
        "Pclass": [3, 1, 3, 1, 3, 3, 3, 2, 3, 1, 3, 2, 1, 1, 1],
        "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    })
    res = train_baseline_model(df, target_col="Survived", problem_type="Classification")
    assert "error" not in res
    assert res["problem_type"] == "Classification"
    assert "Accuracy" in res["metrics"]
    assert "F1_Score" in res["metrics"]
    assert len(res["top_features"]) > 0


def test_train_baseline_regression():
    df = pd.DataFrame({
        "Units": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "Discount": [0.0, 0.05, 0.1, 0.0, 0.15, 0.2, 0.05, 0.1, 0.0, 0.25, 0.0, 0.1, 0.05, 0.2, 0.15],
        "Revenue": [100, 190, 270, 400, 425, 480, 665, 720, 900, 750, 1100, 1080, 1235, 1120, 1275]
    })
    res = train_baseline_model(df, target_col="Revenue", problem_type="Regression")
    assert "error" not in res
    assert res["problem_type"] == "Regression"
    assert "R2_Score" in res["metrics"]
    assert "RMSE" in res["metrics"]
