from typing import Any, Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC

class ClassifierFactory:
    """Creates scikit-learn classification models and advanced ensembles with standardized parameters."""

    @staticmethod
    def get_model(model_name: str, use_class_weights: bool = False, **kwargs) -> BaseEstimator:
        import re
        name = re.sub(r"[^a-zA-Z0-9]", "", model_name.lower())
        cw = "balanced" if use_class_weights else None

        if "logisticregression" in name or name in ["lr"]:
            return LogisticRegression(max_iter=1000, class_weight=cw, random_state=42, **kwargs)
        elif "decisiontree" in name or name in ["dt"]:
            return DecisionTreeClassifier(class_weight=cw, random_state=42, **kwargs)
        elif "randomforest" in name or "bagging" in name or name in ["rf"]:
            return RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1, **kwargs)
        elif "svm" in name or "svc" in name or "kernel" in name:
            return SVC(probability=True, class_weight=cw, random_state=42, **kwargs)
        elif "gradientboosting" in name or "boosting" in name or name in ["gb"]:
            return GradientBoostingClassifier(random_state=42, **kwargs)
        elif "voting" in name:
            estimators = [
                ("lr", LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(random_state=42))
            ]
            return VotingClassifier(estimators=estimators, voting="soft", **kwargs)
        elif "stacking" in name:
            base_estimators = [
                ("dt", DecisionTreeClassifier(class_weight=cw, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=50, class_weight=cw, random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(random_state=42))
            ]
            meta_learner = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
            return StackingClassifier(estimators=base_estimators, final_estimator=meta_learner, cv=5, **kwargs)
        else:
            raise ValueError(f"Unknown classifier model name: {model_name}")

    @staticmethod
    def create_candidates(selected_names: List[str], use_class_weights: bool = False) -> Dict[str, BaseEstimator]:
        """Dynamically instantiates model dictionary from a list of agent-selected algorithm names."""
        candidates = {}
        alias_map = {
            "logistic_regression": "Logistic Regression",
            "lr": "Logistic Regression",
            "decision_tree": "Decision Tree",
            "dt": "Decision Tree",
            "random_forest": "Random Forest (Bagging)",
            "rf": "Random Forest (Bagging)",
            "bagging": "Random Forest (Bagging)",
            "gradient_boosting": "Gradient Boosting (Boosting)",
            "gb": "Gradient Boosting (Boosting)",
            "boosting": "Gradient Boosting (Boosting)",
            "svm": "SVM (Kernel)",
            "svc": "SVM (Kernel)",
            "kernel": "SVM (Kernel)",
            "voting_ensemble": "Voting Ensemble (Soft)",
            "voting": "Voting Ensemble (Soft)",
            "stacking_ensemble": "Stacking Ensemble",
            "stacking": "Stacking Ensemble"
        }

        for raw_name in selected_names:
            clean_key = raw_name.lower().replace(" ", "_").replace("-", "")
            display_name = alias_map.get(clean_key, raw_name)
            try:
                candidates[display_name] = ClassifierFactory.get_model(clean_key, use_class_weights=use_class_weights)
            except ValueError:
                # Fallback to nearest supported model
                candidates[display_name] = ClassifierFactory.get_model("random_forest", use_class_weights=use_class_weights)

        return candidates if candidates else ClassifierFactory.get_default_candidates(use_class_weights=use_class_weights)

    @staticmethod
    def get_default_candidates(use_class_weights: bool = False) -> Dict[str, BaseEstimator]:
        """Returns default suite of baseline classification algorithms."""
        return {
            "Logistic Regression": ClassifierFactory.get_model("logistic_regression", use_class_weights=use_class_weights),
            "Decision Tree": ClassifierFactory.get_model("decision_tree", use_class_weights=use_class_weights),
            "Random Forest (Bagging)": ClassifierFactory.get_model("random_forest", use_class_weights=use_class_weights),
            "Gradient Boosting (Boosting)": ClassifierFactory.get_model("gradient_boosting", use_class_weights=use_class_weights),
            "SVM (Kernel)": ClassifierFactory.get_model("svm", use_class_weights=use_class_weights),
            "Voting Ensemble": ClassifierFactory.get_model("voting_ensemble", use_class_weights=use_class_weights),
            "Stacking Ensemble": ClassifierFactory.get_model("stacking_ensemble", use_class_weights=use_class_weights)
        }
