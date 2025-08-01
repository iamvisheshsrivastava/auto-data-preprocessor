"""Convenience imports for the auto_preprocessor package."""

from .data_preprocessor import DataPreprocessor
from .data_cleaning import (
    remove_outliers,
    handle_missing_values,
    remove_duplicate_rows,
    standardize_column_names,
)
from .feature_engineering import apply_pca, create_polynomial_features

__all__ = [
    "DataPreprocessor",
    "remove_outliers",
    "handle_missing_values",
    "remove_duplicate_rows",
    "standardize_column_names",
    "apply_pca",
    "create_polynomial_features",
]
