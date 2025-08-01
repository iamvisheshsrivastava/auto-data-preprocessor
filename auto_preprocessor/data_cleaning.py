import pandas as pd
import numpy as np

def remove_outliers(df, columns, method="zscore", threshold=3):
    """Remove extreme values from specified columns using z-score or IQR.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    columns : list[str]
        List of columns on which outlier removal should be performed.
    method : {"zscore", "iqr"}, optional
        The method used to detect outliers. ``zscore`` uses a robust z-score
        based on the median absolute deviation, while ``iqr`` relies on the
        IQR rule. Default is ``"zscore"``.
    threshold : float, optional
        Threshold value for detecting outliers. For ``zscore`` this is the
        absolute z-score limit. For ``iqr`` it is the multiplier for the IQR
        bounds. Default is ``3``.
    """

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
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= Q1 - threshold * IQR) & (df[column] <= Q3 + threshold * IQR)]
    return df


def handle_missing_values(df, strategy="mean", columns=None, fill_value=None):
    """Fill NA values in specified columns using a strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    strategy : {"mean", "median", "mode", "constant"}, optional
        The imputation strategy. ``constant`` will replace missing values with
        ``fill_value``. Default is ``"mean"``.
    columns : list[str] | None, optional
        Columns to process. If ``None`` all columns are used.
    fill_value : Any, optional
        Value used when ``strategy="constant"``.
    """

    if columns is None:
        columns = df.columns

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
            df[column] = df[column].fillna(fill_value)
    return df

def remove_duplicate_rows(df):
    """Remove duplicate rows from a DataFrame and reset index."""
    return df.drop_duplicates().reset_index(drop=True)
