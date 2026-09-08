from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)

class EvaluationMetrics:
    """Computes comprehensive evaluation metrics for classification and regression."""

    @staticmethod
    def calculate_classification_metrics(y_true: Any, y_pred: Any, y_prob: Any = None) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        }
        
        # Calculate ROC-AUC if probabilities available
        if y_prob is not None:
            try:
                # Binary case
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                elif y_prob.ndim == 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                metrics["roc_auc"] = 0.5
        else:
            metrics["roc_auc"] = 0.5
            
        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))
        }

    @staticmethod
    def get_confusion_matrix(y_true: Any, y_pred: Any) -> list:
        """Returns 2D confusion matrix as nested lists."""
        return confusion_matrix(y_true, y_pred).tolist()
