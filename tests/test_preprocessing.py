import pandas as pd
import numpy as np
from src.preprocessing.pipeline import PreprocessingPipeline

def test_preprocessing_pipeline():
    df = pd.DataFrame({
        "num1": [1.0, np.nan, 3.0, 4.0],
        "cat1": ["A", "B", "A", np.nan],
        "id_col": ["ID1", "ID2", "ID3", "ID4"]
    })
    
    config = {
        "exclude_columns": ["id_col"],
        "numerical_imputation": "median",
        "scaling": "standard",
        "categorical_encoding": "one_hot"
    }
    
    pipeline = PreprocessingPipeline(config)
    X_proc = pipeline.fit_transform(df)
    
    assert pipeline.fitted
    # Processed shape: 1 numerical + 2 one-hot categorical columns = 3 columns
    assert X_proc.shape[0] == 4
    assert X_proc.shape[1] >= 2
    
    # Check feature names
    names = pipeline.get_feature_names()
    assert "num1" in names
    assert any("cat1" in n for n in names)
    assert "id_col" not in names
