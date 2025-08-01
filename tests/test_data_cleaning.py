import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_preprocessor.data_cleaning import (
    remove_outliers,
    handle_missing_values,
    remove_duplicate_rows,
    drop_high_missing_columns,
)

def test_remove_outliers():
    data = {
        "age": [25, 30, 35, 40, 1000],
        "income": [50000, 60000, 70000, 80000, 999999]
    }
    df = pd.DataFrame(data)
    cleaned_df = remove_outliers(df, ["age", "income"], method="iqr", threshold=1.5)
    assert len(cleaned_df) < len(df)

def test_handle_missing_values():
    data = {
        "age": [25, None, 35, None, 45],
        "income": [50000, 60000, None, 80000, 90000]
    }
    df = pd.DataFrame(data)
    imputed_df = handle_missing_values(df, strategy="mean")
    assert imputed_df.isnull().sum().sum() == 0

def test_remove_duplicate_rows():
    data = {
        "age": [25, 25, 30],
        "income": [50000, 50000, 60000]
    }
    df = pd.DataFrame(data)
    deduped_df = remove_duplicate_rows(df)
    assert len(deduped_df) == 2


def test_drop_high_missing_columns():
    data = {
        "a": [1, None, None],
        "b": [1, 2, 3],
        "c": [None, None, None]
    }
    df = pd.DataFrame(data)
    cleaned = drop_high_missing_columns(df, threshold=0.7)
    assert "c" not in cleaned.columns
    assert "a" in cleaned.columns
