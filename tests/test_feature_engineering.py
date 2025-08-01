import pandas as pd
from auto_preprocessor.feature_engineering import apply_pca, create_polynomial_features

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

def test_create_polynomial_features():
    data = {
        "f1": [1, 2, 3],
        "f2": [4, 5, 6]
    }
    df = pd.DataFrame(data)
    poly_df = create_polynomial_features(df, degree=2)
    # number of output columns should be > input columns for degree>1
    assert poly_df.shape[1] > df.shape[1]
    assert poly_df.shape[0] == len(df)
