import pandas as pd
import numpy as np

def remove_outliers(df, columns, method="zscore", threshold=3):
    if method == "zscore":
        for column in columns:
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            df = df[z_scores.abs() <= threshold]
    elif method == "iqr":
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df

def handle_missing_values(df, strategy="mean", columns=None):
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
    return df
