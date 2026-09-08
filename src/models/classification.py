from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class ClassifierFactory:
    """Creates instances of scikit-learn classification models."""
    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseEstimator:
        name = model_name.lower().replace("_", "").replace(" ", "")
        if name == "logisticregression":
            return LogisticRegression(**kwargs)
        elif name == "decisiontree":
            return DecisionTreeClassifier(**kwargs)
        elif name == "randomforest":
            return RandomForestClassifier(**kwargs)
        elif name == "svm" or name == "svc":
            return SVC(**kwargs)
        elif name == "gradientboosting":
            return GradientBoostingClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier model name: {model_name}")
