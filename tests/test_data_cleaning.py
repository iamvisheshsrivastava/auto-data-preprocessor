import pandas as pd
from auto_preprocessor.data_cleaning import remove_outliers, handle_missing_values
from auto_preprocessor.data_cleaning import remove_duplicate_rows

def test_remove_outliers():
    data = {
        "age": [25, 30, 35, 40, 1000],
        "income": [50000, 60000, 70000, 80000, 999999]
    }
    df = pd.DataFrame(data)
    cleaned_df = remove_outliers(df, ["age", "income"], method="zscore", threshold=3)
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
