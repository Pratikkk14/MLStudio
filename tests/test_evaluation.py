import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.validation import ValidationEngine
from src.evaluation.comparison import ModelComparator

def test_metrics_classification():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    metrics = EvaluationMetrics.calculate_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert "f1" in metrics

def test_metrics_regression():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.0]
    metrics = EvaluationMetrics.calculate_regression_metrics(y_true, y_pred)
    assert metrics["rmse"] > 0
    assert "r2" in metrics

def test_model_comparator():
    comp = ModelComparator("f1")
    comp.add_result({"model_name": "model_a", "metrics": {"f1": 0.8}, "runtime_seconds": 1.2})
    comp.add_result({"model_name": "model_b", "metrics": {"f1": 0.9}, "runtime_seconds": 2.5})
    best = comp.get_best_model()
    assert best["model_name"] == "model_b"

def test_validation_engine_cross_validate():
    X = np.random.randn(30, 4)
    y = np.random.choice([0, 1], size=30)
    
    engine = ValidationEngine(task_type="binary_classification", n_splits=3)
    res = engine.cross_validate(
        model=LogisticRegression(),
        model_name="Logistic Regression",
        X_train=X,
        y_train=y,
        primary_metric="accuracy"
    )
    
    assert res["status"] == "completed"
    assert "accuracy" in res["metrics"]
    assert res["cv_mean"] >= 0
