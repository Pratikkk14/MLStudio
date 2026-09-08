import pytest
import numpy as np
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
    comp.add_result("model_a", {"f1": 0.8}, 1.2)
    comp.add_result("model_b", {"f1": 0.9}, 2.5)
    best = comp.get_best_model()
    assert best["model_name"] == "model_b"
