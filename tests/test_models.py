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
    
    voting = ClassifierFactory.get_model("voting_ensemble")
    assert voting is not None
    
    stacking = ClassifierFactory.get_model("stacking_ensemble")
    assert stacking is not None

def test_classifier_factory_create_candidates():
    candidates = ClassifierFactory.create_candidates(["random_forest", "gradient_boosting", "voting_ensemble"])
    assert len(candidates) == 3
    assert "Random Forest (Bagging)" in candidates
    assert "Voting Ensemble (Soft)" in candidates

def test_classifier_factory_invalid():
    with pytest.raises(ValueError):
        ClassifierFactory.get_model("invalid_model_name")

def test_regressor_factory():
    model = RegressorFactory.get_model("linear_regression")
    assert isinstance(model, LinearRegression)
    
    voting = RegressorFactory.get_model("voting_ensemble")
    assert voting is not None
    
    stacking = RegressorFactory.get_model("stacking_ensemble")
    assert stacking is not None

def test_regressor_factory_create_candidates():
    candidates = RegressorFactory.create_candidates(["decision_tree", "ridge", "stacking_ensemble"])
    assert len(candidates) == 3
    assert "Decision Tree" in candidates
    assert "Stacking Ensemble" in candidates

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
