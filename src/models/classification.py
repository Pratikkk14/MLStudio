from typing import Any, Dict, List
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class ClassifierFactory:
    """Creates scikit-learn classification models with standardized parameters."""

    @staticmethod
    def get_model(model_name: str, use_class_weights: bool = False, **kwargs) -> BaseEstimator:
        name = model_name.lower().replace("_", "").replace(" ", "").replace("-", "")
        cw = "balanced" if use_class_weights else None

        if name in ["logisticregression", "lr"]:
            return LogisticRegression(max_iter=1000, class_weight=cw, random_state=42, **kwargs)
        elif name in ["decisiontree", "dt", "decisiontreeclassifier"]:
            return DecisionTreeClassifier(class_weight=cw, random_state=42, **kwargs)
        elif name in ["randomforest", "rf", "randomforestclassifier"]:
            return RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1, **kwargs)
        elif name in ["svm", "svc"]:
            return SVC(probability=True, class_weight=cw, random_state=42, **kwargs)
        elif name in ["gradientboosting", "gb", "gradientboostingclassifier"]:
            return GradientBoostingClassifier(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown classifier model name: {model_name}")

    @staticmethod
    def get_default_candidates(use_class_weights: bool = False) -> Dict[str, BaseEstimator]:
        """Returns standard suite of baseline classification algorithms."""
        return {
            "Logistic Regression": ClassifierFactory.get_model("logistic_regression", use_class_weights=use_class_weights),
            "Decision Tree": ClassifierFactory.get_model("decision_tree", use_class_weights=use_class_weights),
            "Random Forest": ClassifierFactory.get_model("random_forest", use_class_weights=use_class_weights),
            "Gradient Boosting": ClassifierFactory.get_model("gradient_boosting", use_class_weights=use_class_weights),
            "SVM": ClassifierFactory.get_model("svm", use_class_weights=use_class_weights)
        }
