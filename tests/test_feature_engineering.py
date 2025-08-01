import pandas as pd
from auto_preprocessor.feature_engineering import apply_pca, add_polynomial_features

def test_apply_pca():
    data = {
        "feature1": [1, 2, 3, 4],
        "feature2": [4, 3, 2, 1],
        "feature3": [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)
    components = apply_pca(df, n_components=2)
    assert components.shape[1] == 2
    assert components.shape[0] == len(df)


def test_add_polynomial_features():
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    transformed = add_polynomial_features(df, ["f1", "f2"], degree=2)
    assert "f1^2" in transformed.columns
    assert "f1 f2" in transformed.columns
