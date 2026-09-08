import pandas as pd
from typing import Dict, Any
from src.preprocessing.strategies import NumericalImputationStrategy, ScalingStrategy, CategoricalEncodingStrategy

class PreprocessingPipeline:
    """Manages the creation and execution of data preprocessing steps."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.imputation_strategy = config.get("numerical_imputation", NumericalImputationStrategy.MEDIAN)
        self.scaling_strategy = config.get("scaling", ScalingStrategy.STANDARD)
        self.encoding_strategy = config.get("categorical_encoding", CategoricalEncodingStrategy.ONE_HOT)
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits preprocessing parameters and transforms the dataframe (skeleton)."""
        self.fitted = True
        return df.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a dataframe using already fitted parameters (skeleton)."""
        return df.copy()
