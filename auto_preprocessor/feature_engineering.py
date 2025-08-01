from sklearn.decomposition import PCA


def apply_pca(df, n_components, return_dataframe=False):
    """Apply PCA and return the principal components.

    Parameters
    ----------
    df : pandas.DataFrame
        The input feature matrix.
    n_components : int
        Number of principal components to compute.
    return_dataframe : bool, default False
        If ``True``, a ``pandas.DataFrame`` with appropriately named columns is
        returned instead of a NumPy array.
    """

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)

    if return_dataframe:
        import pandas as pd

        cols = [f"PC{i+1}" for i in range(components.shape[1])]
        return pd.DataFrame(components, columns=cols, index=df.index)
    return components
