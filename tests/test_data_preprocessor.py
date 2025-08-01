import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def test_preprocess_minmax_scaling_and_missing_threshold():
    data = {
        "age": [0, 10, None, 20],
        "income": [100, 200, 300, 400],
        "gender": ["M", "F", "M", "F"],
        "purchased": [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    preprocessor = DataPreprocessor(scaling_strategy="minmax", missing_threshold=0.25)
    X, y = preprocessor.preprocess(df, target_column="purchased")
    assert X["age"].min() >= 0 and X["age"].max() <= 1
    # column 'age' has one missing value < threshold -> kept
    assert "age" in X.columns
