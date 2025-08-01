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


def add_polynomial_features(df, columns, degree=2):
    """Generate polynomial features for the specified columns."""
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    poly_array = pf.fit_transform(df[columns])
    new_cols = pf.get_feature_names_out(columns)
    poly_df = pd.DataFrame(poly_array, columns=new_cols, index=df.index)
    df = df.drop(columns=columns)
    df = pd.concat([df, poly_df], axis=1)
    return df
