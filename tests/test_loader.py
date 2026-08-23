import os
import pytest
import pandas as pd
from insightforge.loader import validate_file, load_dataset


def test_validate_file_existing_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n1,2,3\n")
    res = validate_file(str(csv_file))
    assert res["valid"] is True
    assert res["format"] == "csv"


def test_validate_file_non_existent():
    res = validate_file("non_existent_file_xyz.csv")
    assert res["valid"] is False
    assert "does not exist" in res["message"]


def test_load_dataset_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("col1,col2\n10,20\n30,40\n")
    df, info = load_dataset(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert info["columns"] == 2
