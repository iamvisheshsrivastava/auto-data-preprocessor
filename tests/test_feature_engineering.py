import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_preprocessor.feature_engineering import apply_pca

def test_apply_pca():
    data = {
        "feature1": [1, 2, 3, 4],
        "feature2": [4, 3, 2, 1],
        "feature3": [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)
    components, pca = apply_pca(df, n_components=2, return_dataframe=True)
    assert list(components.columns) == ["pc1", "pc2"]
    assert components.shape[0] == len(df)
    assert hasattr(pca, "components_")
