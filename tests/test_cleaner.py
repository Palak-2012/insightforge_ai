import numpy as np
import pandas as pd
from insightforge.agents.cleaner import clean_dataframe


def test_clean_dataframe_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": ["x", "x", "y"]
    })
    cleaned, report = clean_dataframe(df)
    assert len(cleaned) == 2
    assert report["duplicates_removed"] == 1


def test_clean_dataframe_imputation():
    df = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 30.0],
        "cat": ["Apple", "Apple", np.nan, "Banana"]
    })
    cleaned, report = clean_dataframe(df)
    assert cleaned["num"].isnull().sum() == 0
    assert cleaned["cat"].isnull().sum() == 0
    assert report["nulls_filled_total"] == 2
    assert cleaned["num"].iloc[2] == 20.0  # median
    assert cleaned["cat"].iloc[2] == "Apple"  # mode
