from sklearn.decomposition import PCA


def apply_pca(df, n_components):
    """Apply PCA and return the principal components."""
    # initialize PCA transformer with the requested number of components
    pca = PCA(n_components=n_components)
    # compute the component representation
    components = pca.fit_transform(df)
    return components
