from typing import Any, Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)

class RegressorFactory:
    """Creates scikit-learn regression models and advanced ensembles."""

    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseEstimator:
        import re
        name = re.sub(r"[^a-zA-Z0-9]", "", model_name.lower())

        if "linearregression" in name or name in ["lr"]:
            return LinearRegression(**kwargs)
        elif "ridge" in name:
            return Ridge(random_state=42, **kwargs)
        elif "decisiontree" in name or name in ["dt"]:
            return DecisionTreeRegressor(random_state=42, **kwargs)
        elif "randomforest" in name or "bagging" in name or name in ["rf"]:
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **kwargs)
        elif "gradientboosting" in name or "boosting" in name or name in ["gb"]:
            return GradientBoostingRegressor(random_state=42, **kwargs)
        elif "voting" in name:
            estimators = [
                ("ridge", Ridge(random_state=42)),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingRegressor(random_state=42))
            ]
            return VotingRegressor(estimators=estimators, **kwargs)
        elif "stacking" in name:
            base_estimators = [
                ("dt", DecisionTreeRegressor(random_state=42)),
                ("rf", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingRegressor(random_state=42))
            ]
            meta_learner = Ridge(random_state=42)
            return StackingRegressor(estimators=base_estimators, final_estimator=meta_learner, cv=5, **kwargs)
        else:
            raise ValueError(f"Unknown regressor model name: {model_name}")

    @staticmethod
    def create_candidates(selected_names: List[str]) -> Dict[str, BaseEstimator]:
        """Dynamically instantiates regression model dictionary from a list of agent-selected algorithm names."""
        candidates = {}
        alias_map = {
            "linear_regression": "Linear Regression",
            "lr": "Linear Regression",
            "ridge": "Ridge Regression",
            "decision_tree": "Decision Tree",
            "dt": "Decision Tree",
            "random_forest": "Random Forest (Bagging)",
            "rf": "Random Forest (Bagging)",
            "bagging": "Random Forest (Bagging)",
            "gradient_boosting": "Gradient Boosting (Boosting)",
            "gb": "Gradient Boosting (Boosting)",
            "boosting": "Gradient Boosting (Boosting)",
            "voting_ensemble": "Voting Ensemble",
            "voting": "Voting Ensemble",
            "stacking_ensemble": "Stacking Ensemble",
            "stacking": "Stacking Ensemble"
        }

        for raw_name in selected_names:
            clean_key = raw_name.lower().replace(" ", "_").replace("-", "")
            display_name = alias_map.get(clean_key, raw_name)
            try:
                candidates[display_name] = RegressorFactory.get_model(clean_key)
            except ValueError:
                candidates[display_name] = RegressorFactory.get_model("random_forest")

        return candidates if candidates else RegressorFactory.get_default_candidates()

    @staticmethod
    def get_default_candidates() -> Dict[str, BaseEstimator]:
        """Returns standard suite of baseline regression algorithms."""
        return {
            "Linear Regression": RegressorFactory.get_model("linear_regression"),
            "Ridge Regression": RegressorFactory.get_model("ridge"),
            "Decision Tree": RegressorFactory.get_model("decision_tree"),
            "Random Forest (Bagging)": RegressorFactory.get_model("random_forest"),
            "Gradient Boosting (Boosting)": RegressorFactory.get_model("gradient_boosting"),
            "Voting Ensemble": RegressorFactory.get_model("voting_ensemble"),
            "Stacking Ensemble": RegressorFactory.get_model("stacking_ensemble")
        }
