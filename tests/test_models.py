import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from src.models.classification import ClassifierFactory
from src.models.regression import RegressorFactory
from src.models.tuning import HyperparameterTuner

def test_classifier_factory():
    model = ClassifierFactory.get_model("logistic_regression")
    assert isinstance(model, LogisticRegression)

def test_classifier_factory_invalid():
    with pytest.raises(ValueError):
        ClassifierFactory.get_model("invalid_model_name")

def test_regressor_factory():
    model = RegressorFactory.get_model("linear_regression")
    assert isinstance(model, LinearRegression)

def test_hyperparameter_tuning():
    X = np.random.randn(30, 4)
    y = np.random.choice([0, 1], size=30)
    
    base_model = RandomForestClassifier(random_state=42)
    res = HyperparameterTuner.tune(
        model=base_model,
        model_name="random_forest",
        X=X,
        y=y,
        task_type="binary_classification",
        scoring_metric="accuracy",
        n_iter=2
    )
    
    assert "best_estimator" in res
    assert res["trials_run"] >= 1
    assert res["best_score"] > 0
