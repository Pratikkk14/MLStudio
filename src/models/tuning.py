import time
from typing import Any, Dict, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, StratifiedKFold

class HyperparameterTuner:
    """Performs cross-validated hyperparameter optimization for candidate models."""

    @staticmethod
    def get_param_grid(model_name: str) -> Dict[str, list]:
        """Returns standard tuning search space for candidate models."""
        name = model_name.lower().replace("_", "").replace(" ", "").replace("-", "")
        
        if "randomforest" in name or name == "rf":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif "gradientboosting" in name or name == "gb":
            return {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            }
        elif "logisticregression" in name or name == "lr":
            return {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"]
            }
        elif "decisiontree" in name or name == "dt":
            return {
                "max_depth": [3, 5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif "svm" in name or "svc" in name:
            return {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"]
            }
        else:
            return {}

    @classmethod
    def tune(
        cls,
        model: BaseEstimator,
        model_name: str,
        X: Any,
        y: Any,
        task_type: str,
        scoring_metric: str,
        n_iter: int = 15,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Tunes hyperparameters using RandomizedSearchCV."""
        param_grid = cls.get_param_grid(model_name)
        if not param_grid:
            # If no parameter grid defined, return fitted original model
            model.fit(X, y)
            return {
                "best_estimator": model,
                "best_params": {},
                "best_score": 0.0,
                "trials_run": 1,
                "tuning_time_seconds": 0.0
            }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if "classification" in task_type else KFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # Scikit-learn scoring string mapping
        scoring_map = {
            "f1": "f1_macro",
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "roc_auc": "roc_auc",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
        sk_scoring = scoring_map.get(scoring_metric.lower(), "f1_macro" if "classification" in task_type else "r2")

        # Fallback to accuracy if roc_auc fails on multiclass
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=min(n_iter, 20),
            scoring=sk_scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            error_score="raise"
        )

        start_t = time.time()
        search.fit(X, y)
        runtime = time.time() - start_t

        best_score = abs(search.best_score_)  # abs handles negative MAE/RMSE
        
        return {
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": float(best_score),
            "trials_run": len(search.cv_results_["params"]),
            "tuning_time_seconds": float(runtime)
        }
