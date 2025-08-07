from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def apply_pca(df, n_components, return_dataframe=False):
    """Apply PCA and return the principal components and fitted PCA."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    if return_dataframe:
        cols = [f"pc{i+1}" for i in range(components.shape[1])]
        components = pd.DataFrame(components, columns=cols, index=df.index)
    return components, pca


def generate_polynomial_features(df, degree=2, include_bias=False, return_dataframe=True):
    """
    Generate polynomial and interaction features for the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input features.
    degree : int, default=2
        The maximum degree of the polynomial features.
    include_bias : bool, default=False
        If True, include a bias column (all ones) in the output.
    return_dataframe : bool, default=True
        Whether to return the transformed features as a pandas DataFrame.

    Returns
    -------
    transformed : numpy.ndarray or pandas.DataFrame
        The polynomial features. Returned as a DataFrame if ``return_dataframe``
        is True, otherwise as a NumPy array.
    poly : sklearn.preprocessing.PolynomialFeatures
        The fitted PolynomialFeatures instance.
    """

    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    transformed = poly.fit_transform(df)
    if return_dataframe:
        feature_names = poly.get_feature_names_out(df.columns)
        transformed = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    return transformed, poly
