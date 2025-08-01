from sklearn.decomposition import PCA
import pandas as pd


def apply_pca(df, n_components, return_dataframe=False):
    """Apply PCA and return the principal components and fitted PCA."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    if return_dataframe:
        cols = [f"pc{i+1}" for i in range(components.shape[1])]
        components = pd.DataFrame(components, columns=cols, index=df.index)
    return components, pca
