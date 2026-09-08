import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
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

def test_tuner_skeleton():
    model = LinearRegression()
    tuner = HyperparameterTuner(model, {})
    res = tuner.tune(None, None)
    assert res["best_score"] == 0.0
    assert res["trials_run"] == 20
