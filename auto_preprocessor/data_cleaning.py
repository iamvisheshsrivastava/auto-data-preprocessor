import pandas as pd
import numpy as np

def remove_outliers(df, columns, method="zscore", threshold=3):
    """Remove extreme values from specified columns using z-score or IQR."""
    if method == "zscore":
        for column in columns:
            # compute z-scores for the column
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            # keep rows within the threshold
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


def handle_missing_values(df, strategy="mean", columns=None):
    """Handle NA values in specified columns using a given strategy.

    Supported strategies are "mean", "median", "mode" and "drop".
    When ``columns`` is ``None`` all columns are processed.
    """
    if columns is None:
        columns = df.columns  # default to all columns
    if strategy == "mean":
        for column in columns:
            # fill missing values with column mean
            df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        for column in columns:
            # fill missing values with column median
            df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        for column in columns:
            # fill missing values with column mode
            df[column] = df[column].fillna(df[column].mode()[0])
    elif strategy == "drop":
        df = df.dropna(subset=columns)
    return df

def remove_duplicate_rows(df, subset=None):
    """Remove duplicate rows from a DataFrame and reset index.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    subset : list or None
        Optional subset of columns to consider when identifying duplicates.
    """
    return df.drop_duplicates(subset=subset).reset_index(drop=True)
