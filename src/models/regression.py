from typing import Any
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class RegressorFactory:
    """Creates instances of scikit-learn regression models."""
    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseEstimator:
        name = model_name.lower().replace("_", "").replace(" ", "")
        if name == "linearregression":
            return LinearRegression(**kwargs)
        elif name == "decisiontree" or name == "decisiontreeregressor":
            return DecisionTreeRegressor(**kwargs)
        elif name == "randomforest" or name == "randomforestregressor":
            return RandomForestRegressor(**kwargs)
        elif name == "gradientboosting" or name == "gradientboostingregressor":
            return GradientBoostingRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown regressor model name: {model_name}")
