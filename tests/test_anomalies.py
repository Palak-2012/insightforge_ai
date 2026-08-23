import pandas as pd
from insightforge.advanced.anomaly_detection import compute_iqr_bounds, detect_anomalies


def test_compute_iqr_bounds():
    s = pd.Series([10, 12, 14, 15, 16, 18, 20, 100])  # 100 is outlier
    bounds = compute_iqr_bounds(s)
    assert bounds["lower_bound"] < 10
    assert bounds["upper_bound"] < 100


def test_detect_anomalies():
    df = pd.DataFrame({"values": [10, 12, 11, 13, 12, 10, 11, 1000]})
    res = detect_anomalies(df)
    assert "values" in res
    assert res["values"]["outlier_count"] == 1
    assert res["values"]["max_outlier"] == 1000.0
