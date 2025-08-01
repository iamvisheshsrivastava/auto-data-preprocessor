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
        "x": [1, 2, 3],
        "y": [4, 5, 6],
    }
    df = pd.DataFrame(data)
    poly_df = add_polynomial_features(df, degree=2)
    # Expect columns for x, y, x^2, x y, y^2
    expected_cols = {"x", "y", "x^2", "x y", "y^2"}
    assert expected_cols.issubset(set(poly_df.columns))
