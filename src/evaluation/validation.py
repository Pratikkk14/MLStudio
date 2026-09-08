import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.base import clone, BaseEstimator
from src.evaluation.metrics import EvaluationMetrics

class ValidationEngine:
    """Manages holdout splitting and K-Fold cross-validation routines."""

    def __init__(self, task_type: str, test_size: float = 0.2, n_splits: int = 5, random_seed: int = 42):
        self.task_type = task_type
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_seed = random_seed

    def split(self, X: Any, y: Any) -> Tuple[Any, Any, Any, Any]:
        """Partitions dataset into Train and untouched Test holdouts."""
        stratify = y if "classification" in self.task_type.lower() and len(np.unique(y)) > 1 else None
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=stratify
        )

    def cross_validate(
        self,
        model: BaseEstimator,
        model_name: str,
        X_train: Any,
        y_train: Any,
        primary_metric: str
    ) -> Dict[str, Any]:
        """Runs Stratified/Standard K-Fold CV and collects metrics per fold."""
        if "classification" in self.task_type.lower():
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            is_classification = True
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            is_classification = False

        fold_metrics = []
        start_time = time.time()

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)

            if is_classification:
                y_prob = None
                if hasattr(fold_model, "predict_proba"):
                    try:
                        y_prob = fold_model.predict_proba(X_val)
                    except Exception:
                        pass
                m = EvaluationMetrics.calculate_classification_metrics(y_val, y_pred, y_prob)
            else:
                m = EvaluationMetrics.calculate_regression_metrics(y_val, y_pred)
                
            fold_metrics.append(m)

        runtime = time.time() - start_time

        # Calculate average metrics and std for primary metric
        avg_metrics = {}
        for k in fold_metrics[0].keys():
            vals = [fm[k] for fm in fold_metrics]
            avg_metrics[k] = float(np.mean(vals))

        primary_scores = [fm.get(primary_metric.lower(), 0.0) for fm in fold_metrics]
        cv_mean = float(np.mean(primary_scores))
        cv_std = float(np.std(primary_scores))

        return {
            "model_name": model_name,
            "metrics": avg_metrics,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "runtime_seconds": float(runtime),
            "status": "completed"
        }
