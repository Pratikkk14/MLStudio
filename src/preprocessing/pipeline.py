from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from src.preprocessing.strategies import NumericalImputationStrategy, ScalingStrategy, CategoricalEncodingStrategy

class PreprocessingPipeline:
    """End-to-end scikit-learn preprocessing pipeline for tabular data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.num_impute_strat = self.config.get("numerical_imputation", "median")
        self.scaling_strat = self.config.get("scaling", "standard")
        self.cat_encode_strat = self.config.get("categorical_encoding", "one_hot")
        self.exclude_columns = set(self.config.get("exclude_columns", []))
        
        self.column_transformer: Optional[ColumnTransformer] = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names_: List[str] = []
        self.fitted = False

    def _build_transformers(self, X: pd.DataFrame) -> ColumnTransformer:
        """Constructs scikit-learn ColumnTransformer based on detected column types."""
        feature_cols = [c for c in X.columns if c not in self.exclude_columns]
        
        self.numerical_cols = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c].dtype)
        ]
        self.categorical_cols = [
            c for c in feature_cols if c not in self.numerical_cols
        ]
        
        transformers = []
        
        # Numerical branch
        if self.numerical_cols:
            num_steps = []
            # Imputation
            num_steps.append(("imputer", SimpleImputer(strategy=self.num_impute_strat)))
            
            # Scaling
            if self.scaling_strat == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif self.scaling_strat == "robust":
                num_steps.append(("scaler", RobustScaler()))
            elif self.scaling_strat == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            # 'none' skips scaler
            
            transformers.append(("num", Pipeline(num_steps), self.numerical_cols))

        # Categorical branch
        if self.categorical_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ]
            if self.cat_encode_strat == "one_hot":
                cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
            elif self.cat_encode_strat == "ordinal":
                cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
                
            transformers.append(("cat", Pipeline(cat_steps), self.categorical_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PreprocessingPipeline":
        """Fits the internal column transformer on training features."""
        self.column_transformer = self._build_transformers(X)
        self.column_transformer.fit(X, y)
        self.fitted = True
        self._extract_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame into processed numpy array."""
        if not self.fitted or self.column_transformer is None:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        return self.column_transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> np.ndarray:
        """Fits and transforms input DataFrame."""
        return self.fit(X, y).transform(X)

    def _extract_feature_names(self) -> None:
        """Extracts transformed column names."""
        names = []
        if not self.column_transformer:
            return
            
        for name, trans, cols in self.column_transformer.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder = trans.named_steps.get("encoder")
                if isinstance(encoder, OneHotEncoder):
                    names.extend(list(encoder.get_feature_names_out(cols)))
                else:
                    names.extend(cols)
        self.feature_names_ = names

    def get_feature_names(self) -> List[str]:
        """Returns list of transformed feature names."""
        return self.feature_names_
