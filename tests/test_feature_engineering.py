import pandas as pd
from auto_preprocessor.feature_engineering import apply_pca, generate_polynomial_features

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


def test_generate_polynomial_features():
    data = {
        "x": [0, 1, 2],
        "y": [1, 2, 3],
    }
    df = pd.DataFrame(data)
    poly_df = generate_polynomial_features(df, degree=2, include_bias=False)
    # number of output columns for 2 features and degree=2 without bias is 5: x, y, x^2, x*y, y^2
    assert poly_df.shape == (3, 5)
