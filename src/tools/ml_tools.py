from typing import Any, Dict, List, Tuple
import pandas as pd
from src.data.loader import DatasetLoader
from src.data.validator import DatasetValidator
from src.data.profiler import DatasetProfiler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.evaluation.validation import ValidationEngine
from src.evaluation.comparison import ModelComparator

class MLTools:
    """Consolidated execution layer tools invoked by the reasoning agent."""
    
    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        loader = DatasetLoader(file_path)
        return loader.load()

    @staticmethod
    def validate_dataset(df: pd.DataFrame, target: str = None) -> Tuple[List[str], List[str]]:
        validator = DatasetValidator(df, target)
        return validator.validate()

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        profiler = DatasetProfiler(df)
        summary = profiler.get_summary()
        col_profiles = profiler.profile_columns()
        return {
            "summary": summary,
            "columns": col_profiles
        }

    @staticmethod
    def detect_target_candidates(df: pd.DataFrame) -> List[str]:
        # Basic heuristic: non-float columns with low cardinality or specific target-like names
        candidates = []
        for col in df.columns:
            name_lower = col.lower()
            if "target" in name_lower or "label" in name_lower or "class" in name_lower or "churn" in name_lower:
                candidates.insert(0, col)
                continue
            
            nunique = df[col].nunique()
            if nunique > 1 and nunique <= 20:
                candidates.append(col)
        return candidates if candidates else list(df.columns[-1:])

    @staticmethod
    def detect_problem_type(df: pd.DataFrame, target: str) -> str:
        nunique = df[target].nunique()
        dtype = str(df[target].dtype)
        if "float" in dtype:
            return "regression"
        elif nunique == 2:
            return "binary_classification"
        else:
            return "multiclass_classification"

    @staticmethod
    def build_preprocessing_pipeline(config: Dict[str, Any]) -> PreprocessingPipeline:
        return PreprocessingPipeline(config)
