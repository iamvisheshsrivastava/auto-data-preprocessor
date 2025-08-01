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
    data = {
        "feature1": [1, 2],
        "feature2": [3, 4]
    }
    df = pd.DataFrame(data)
    poly_df = add_polynomial_features(df, degree=2)
    # columns: 1, f1, f2, f1^2, f1 f2, f2^2 -> 5 if bias excluded? Wait include_bias=False by default -> features: f1, f2, f1^2, f1 f2, f2^2 -> 5
    assert poly_df.shape[1] == 5
    assert poly_df.shape[0] == len(df)
