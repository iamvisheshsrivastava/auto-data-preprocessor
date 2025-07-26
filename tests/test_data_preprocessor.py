import pandas as pd
from auto_preprocessor.data_preprocessor import DataPreprocessor

def test_preprocess():
    data = {
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "gender": ["Male", "Female", "Male", "Female"],
        "purchased": [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df, target_column="purchased")
    assert X.shape[1] == len(data) - 1
    assert len(y) == len(df)

def test_preprocess_onehot_encoding():
    data = {
        "age": [25, 30],
        "income": [50000, 60000],
        "gender": ["Male", "Female"],
        "purchased": [1, 0]
    }
    df = pd.DataFrame(data)
    preprocessor = DataPreprocessor(encoding_strategy="onehot")
    X, y = preprocessor.preprocess(df, target_column="purchased")
    assert "gender_Female" in X.columns
    assert "gender_Male" in X.columns
