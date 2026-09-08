import pandas as pd
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.strategies import NumericalImputationStrategy

def test_preprocessing_pipeline_initialization():
    config = {"numerical_imputation": NumericalImputationStrategy.MEAN}
    pipeline = PreprocessingPipeline(config)
    assert pipeline.imputation_strategy == NumericalImputationStrategy.MEAN
    assert not pipeline.fitted

def test_preprocessing_pipeline_transform():
    df = pd.DataFrame({"col": [1, 2, 3]})
    pipeline = PreprocessingPipeline({})
    df_transformed = pipeline.fit_transform(df)
    assert pipeline.fitted
    pd.testing.assert_frame_equal(df, df_transformed)
