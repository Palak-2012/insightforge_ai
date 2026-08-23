import pandas as pd
from insightforge.agents.eda import (
    get_basic_summary,
    get_statistical_summary,
    get_correlation_matrix,
    get_value_counts,
)


def test_basic_summary():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    res = get_basic_summary(df)
    assert res["rows"] == 3
    assert res["columns"] == 2
    assert "A" in res["column_names"]


def test_statistical_summary():
    df = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0, 50.0]})
    res = get_statistical_summary(df)
    assert "A" in res
    assert res["A"]["mean"] == 30.0
    assert res["A"]["median"] == 30.0
    assert res["A"]["min"] == 10.0
    assert res["A"]["max"] == 50.0


def test_correlation_matrix():
    df = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [2, 4, 6, 8]})
    res = get_correlation_matrix(df)
    assert "X" in res
    assert res["X"]["Y"] == 1.0
