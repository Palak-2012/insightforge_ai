"""
InsightForge AI — Data Loader
=============================
Handles loading and validation of CSV and Excel tabular datasets.
"""

import os
from typing import Any, Dict, Optional, Tuple
import pandas as pd


def validate_file(filepath: str) -> Dict[str, Any]:
    """
    Validates a file path before attempting to load it.

    Parameters
    ----------
    filepath : str — full path to the data file

    Returns
    -------
    dict containing:
        valid    : bool — whether file can be loaded
        format   : str — 'csv', 'excel', or 'unsupported'
        size_mb  : float — file size in megabytes
        message  : str — human-readable validation summary
    """
    if not os.path.exists(filepath):
        return {
            "valid": False,
            "format": "unknown",
            "size_mb": 0.0,
            "message": f"File does not exist: {filepath}"
        }

    ext = os.path.splitext(filepath)[1].lower()
    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)

    if ext == ".csv":
        file_format = "csv"
    elif ext in [".xlsx", ".xls"]:
        file_format = "excel"
    else:
        return {
            "valid": False,
            "format": "unsupported",
            "size_mb": size_mb,
            "message": f"Unsupported format '{ext}'. Use .csv, .xlsx, or .xls"
        }

    return {
        "valid": True,
        "format": file_format,
        "size_mb": size_mb,
        "message": f"Valid {file_format.upper()} file ({size_mb} MB)"
    }


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Loads a CSV file with automatic encoding fallback."""
    for encoding in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            return pd.read_csv(filepath, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV file '{filepath}' with standard encodings.")


def load_excel(filepath: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Loads an Excel file (.xlsx, .xls) with sheet selection."""
    return pd.read_excel(filepath, sheet_name=sheet_name or 0, **kwargs)


def load_dataset(filepath: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified entrypoint to validate and load any supported tabular data file.

    Returns
    -------
    (df, info_dict)
    """
    validation = validate_file(filepath)
    if not validation["valid"]:
        raise ValueError(validation["message"])

    if validation["format"] == "csv":
        df = load_csv(filepath)
    elif validation["format"] == "excel":
        df = load_excel(filepath, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unhandled format: {validation['format']}")

    info = {
        "filepath": filepath,
        "format": validation["format"],
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    return df, info
