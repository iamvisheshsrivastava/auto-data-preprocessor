from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def apply_pca(df, n_components):
    """Apply PCA and return the principal components."""
    # initialize PCA transformer with the requested number of components
    pca = PCA(n_components=n_components)
    # compute the component representation
    components = pca.fit_transform(df)
    return components


def add_polynomial_features(df, degree=2, include_bias=False):
    """Generate polynomial and interaction features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input feature matrix.
    degree : int, default=2
        The polynomial degree.
    include_bias : bool, default=False
        Whether to include a bias (all-ones) column.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the new polynomial features.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    transformed = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)
    return pd.DataFrame(transformed, columns=feature_names, index=df.index)
