import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


def apply_pca(df, n_components):
    """Apply PCA and return the principal components."""
    # initialize PCA transformer with the requested number of components
    pca = PCA(n_components=n_components)
    # compute the component representation
    components = pca.fit_transform(df)
    return components


def add_polynomial_features(df, degree=2, include_bias=False):
    """Generate polynomial features from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input features.
    degree : int, optional
        Degree of the polynomial features. Default is ``2``.
    include_bias : bool, optional
        Whether to include a bias column of ones. Default ``False``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the generated polynomial features with
        appropriately named columns.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_array = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)
    return pd.DataFrame(poly_array, columns=feature_names, index=df.index)
