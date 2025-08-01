"""Top level package for ``auto_preprocessor``.

This module exposes the most commonly used classes and functions so that they
can be conveniently imported from ``auto_preprocessor`` directly.
"""

from .data_cleaning import (
    remove_outliers,
    handle_missing_values,
    remove_duplicate_rows,
)
from .data_preprocessor import DataPreprocessor
from .feature_engineering import apply_pca, add_polynomial_features

__all__ = [
    "remove_outliers",
    "handle_missing_values",
    "remove_duplicate_rows",
    "DataPreprocessor",
    "apply_pca",
    "add_polynomial_features",
]

