from sklearn.decomposition import PCA

def apply_pca(df, n_components):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    return components
