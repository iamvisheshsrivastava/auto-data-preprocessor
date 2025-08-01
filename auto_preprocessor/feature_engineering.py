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


def create_polynomial_features(df, degree):
    """Generate polynomial features of specified degree."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_array = poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)
    return pd.DataFrame(poly_array, columns=feature_names, index=df.index)
