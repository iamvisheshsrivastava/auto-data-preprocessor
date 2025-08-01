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
    """Fill NA values in specified columns using a strategy."""
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
    return df

def remove_duplicate_rows(df):
    """Remove duplicate rows from a DataFrame and reset index."""
    return df.drop_duplicates().reset_index(drop=True)


def drop_high_missing_columns(df, threshold=0.5):
    """Drop columns whose fraction of missing values exceeds the threshold."""
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=cols_to_drop)
