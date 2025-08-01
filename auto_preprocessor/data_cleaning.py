import pandas as pd
import numpy as np

def remove_outliers(df, columns, method="zscore", threshold=3):
    """Remove extreme values from specified columns using z-score or IQR.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to operate on.
    columns : list[str]
        The columns on which to perform outlier removal.
    method : {"zscore", "iqr"}, default "zscore"
        The strategy used to detect outliers.
    threshold : float, default 3
        The cut-off threshold for the chosen method.

    Returns
    -------
    pandas.DataFrame
        A dataframe with outliers removed and the index reset.
    """

    if method == "zscore":
        # Use a robust z-score based on the median and MAD. This works better
        # for small data sets or when extreme outliers skew the standard
        # deviation.
        for column in columns:
            median = df[column].median()
            mad = np.median(np.abs(df[column] - median))
            if mad == 0:
                continue
            modified_z = 0.6745 * (df[column] - median) / mad
            df = df[modified_z.abs() <= threshold]
    elif method == "iqr":
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df.reset_index(drop=True)


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
