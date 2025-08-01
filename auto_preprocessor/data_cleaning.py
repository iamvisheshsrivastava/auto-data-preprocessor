import pandas as pd
import numpy as np

def remove_outliers(df, columns, method="zscore", threshold=3):
    """Remove extreme values from specified columns using z-score or IQR."""
    if method == "zscore":
        for column in columns:
            median = df[column].median()
            mad = np.median(np.abs(df[column] - median))
            if mad == 0:
                mad = 1e-9
            z_scores = 0.6745 * (df[column] - median) / mad
            df = df[z_scores.abs() <= threshold]
    elif method == "iqr":
        for column in columns:
            # compute the 25th percentile
            Q1 = df[column].quantile(0.25)
            # compute the 75th percentile
            Q3 = df[column].quantile(0.75)
            # interquartile range
            IQR = Q3 - Q1
            # filter rows within 1.5 * IQR bounds
            df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df


def handle_missing_values(df, strategy="mean", columns=None, constant_value=None):
    """Fill NA values in specified columns using a strategy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to operate on.
    strategy : str, default "mean"
        Filling strategy ("mean", "median", "mode", "constant").
    columns : list, optional
        Subset of columns to apply the strategy on.
    constant_value : any, optional
        Value used when ``strategy`` is ``"constant"``.
    """

    if columns is None:
        columns = df.columns  # default to all columns

    if strategy == "mean":
        for column in columns:
            df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        for column in columns:
            df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        for column in columns:
            df[column] = df[column].fillna(df[column].mode()[0])
    elif strategy == "constant":
        for column in columns:
            df[column] = df[column].fillna(constant_value)

    return df


def normalize_column_names(df):
    """Standardize column names to snake_case.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be renamed.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names.
    """
    mapping = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    return df.rename(columns=mapping)

def remove_duplicate_rows(df):
    """Remove duplicate rows from a DataFrame and reset index."""
    return df.drop_duplicates().reset_index(drop=True)
