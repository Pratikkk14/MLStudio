import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from src.data.loader import DatasetLoader
from src.data.validator import DatasetValidator
from src.data.profiler import DatasetProfiler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.classification import ClassifierFactory
from src.models.regression import RegressorFactory
from src.models.tuning import HyperparameterTuner
from src.evaluation.validation import ValidationEngine
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.comparison import ModelComparator

class MLTools:
    """Consolidated deterministic tools called by the reasoning agent and workflow runner."""

    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        return DatasetLoader(file_path).load()

    @staticmethod
    def validate_dataset(df: pd.DataFrame, target: Optional[str] = None) -> Tuple[List[str], List[str]]:
        return DatasetValidator(df, target).validate()

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        profiler = DatasetProfiler(df)
        return {
            "summary": profiler.get_summary(),
            "columns": profiler.profile_columns()
        }

    @staticmethod
    def detect_target_candidates(df: pd.DataFrame) -> List[str]:
        candidates = []
        for col in df.columns:
            name_lower = col.lower()
            if any(kw in name_lower for kw in ["target", "label", "class", "churn", "status", "outcome"]):
                candidates.insert(0, col)
                continue
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= 20:
                candidates.append(col)
        return candidates if candidates else list(df.columns[-1:])

    @staticmethod
    def detect_problem_type(df: pd.DataFrame, target: str) -> str:
        target_series = df[target].dropna()
        nunique = target_series.nunique()
        if pd.api.types.is_float_dtype(target_series.dtype) and nunique > 20:
            return "regression"
        elif nunique == 2:
            return "binary_classification"
        else:
            return "multiclass_classification"

    @staticmethod
    def build_preprocessing_pipeline(config: Dict[str, Any]) -> PreprocessingPipeline:
        return PreprocessingPipeline(config)

    @staticmethod
    def run_baseline_experiments(
        task_type: str,
        X_train: Any,
        y_train: Any,
        primary_metric: str,
        use_class_weights: bool = False
    ) -> List[Dict[str, Any]]:
        """Trains and cross-validates candidate baseline models."""
        if "classification" in task_type.lower():
            candidates = ClassifierFactory.get_default_candidates(use_class_weights=use_class_weights)
        else:
            candidates = RegressorFactory.get_default_candidates()

        validator = ValidationEngine(task_type=task_type, n_splits=5)
        results = []

        for name, model in candidates.items():
            record = validator.cross_validate(model, name, X_train, y_train, primary_metric)
            results.append(record)

        return results

    @staticmethod
    def tune_candidate(
        model_name: str,
        X_train: Any,
        y_train: Any,
        task_type: str,
        primary_metric: str,
        trials: int = 15,
        use_class_weights: bool = False
    ) -> Dict[str, Any]:
        """Runs hyperparameter search on candidate model."""
        if "classification" in task_type.lower():
            base_model = ClassifierFactory.get_model(model_name, use_class_weights=use_class_weights)
        else:
            base_model = RegressorFactory.get_model(model_name)

        return HyperparameterTuner.tune(
            model=base_model,
            model_name=model_name,
            X=X_train,
            y=y_train,
            task_type=task_type,
            scoring_metric=primary_metric,
            n_iter=trials
        )

    @staticmethod
    def evaluate_holdout_test(
        model: Any,
        X_test: Any,
        y_test: Any,
        task_type: str
    ) -> Tuple[Dict[str, float], Optional[list]]:
        """Evaluates final fitted model on holdout test set."""
        y_pred = model.predict(X_test)
        if "classification" in task_type.lower():
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass
            metrics = EvaluationMetrics.calculate_classification_metrics(y_test, y_pred, y_prob)
            cm = EvaluationMetrics.get_confusion_matrix(y_test, y_pred)
            return metrics, cm
        else:
            metrics = EvaluationMetrics.calculate_regression_metrics(y_test, y_pred)
            return metrics, None

    @staticmethod
    def extract_feature_importance(model: Any, feature_names: List[str]) -> List[Tuple[str, float]]:
        """Extracts top feature importances or linear coefficients."""
        importances = []
        
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            for name, score in zip(feature_names, fi):
                importances.append((name, float(score)))
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            for name, score in zip(feature_names, coef):
                importances.append((name, float(score)))
        else:
            for name in feature_names:
                importances.append((name, 1.0 / max(len(feature_names), 1)))

        importances.sort(key=lambda x: x[1], reverse=True)
        return importances

    @staticmethod
    def export_artifacts(
        state_dict: Dict[str, Any],
        best_model: Any,
        final_report: Dict[str, Any],
        report_text: str,
        out_dir: str = "outputs"
    ) -> None:
        """Saves all generated deliverables into outputs/ directory."""
        p_out = Path(out_dir)
        (p_out / "reports").mkdir(parents=True, exist_ok=True)
        (p_out / "experiments").mkdir(parents=True, exist_ok=True)
        (p_out / "models").mkdir(parents=True, exist_ok=True)

        with open(p_out / "reports" / "final_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, default=str)
        with open(p_out / "reports" / "final_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        with open(p_out / "experiments" / "experiment_history.json", "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

        joblib.dump(best_model, p_out / "models" / "best_model.joblib")
