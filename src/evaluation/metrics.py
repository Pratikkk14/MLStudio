from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

class EvaluationMetrics:
    """Calculates model metrics for classification and regression tasks."""
    @staticmethod
    def calculate_classification_metrics(y_true: Any, y_pred: Any, y_prob: Any = None) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                metrics["roc_auc"] = 0.5
        else:
            metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = 0.5  # Placeholder for PR-AUC
        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))
        }
