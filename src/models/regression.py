from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class RegressorFactory:
    """Creates scikit-learn regression models."""

    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseEstimator:
        name = model_name.lower().replace("_", "").replace(" ", "").replace("-", "")

        if name in ["linearregression", "lr"]:
            return LinearRegression(**kwargs)
        elif name in ["ridge"]:
            return Ridge(random_state=42, **kwargs)
        elif name in ["decisiontree", "decisiontreeregressor", "dt"]:
            return DecisionTreeRegressor(random_state=42, **kwargs)
        elif name in ["randomforest", "randomforestregressor", "rf"]:
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **kwargs)
        elif name in ["gradientboosting", "gradientboostingregressor", "gb"]:
            return GradientBoostingRegressor(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown regressor model name: {model_name}")

    @staticmethod
    def get_default_candidates() -> Dict[str, BaseEstimator]:
        """Returns standard suite of baseline regression algorithms."""
        return {
            "Linear Regression": RegressorFactory.get_model("linear_regression"),
            "Decision Tree": RegressorFactory.get_model("decision_tree"),
            "Random Forest": RegressorFactory.get_model("random_forest"),
            "Gradient Boosting": RegressorFactory.get_model("gradient_boosting")
        }
