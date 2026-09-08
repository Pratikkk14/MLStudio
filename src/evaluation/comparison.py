from typing import Any, Dict, List

class ModelComparator:
    """Aggregates and compares candidate model experiment results."""

    def __init__(self, primary_metric: str):
        self.primary_metric = primary_metric.lower()
        self.results: List[Dict[str, Any]] = []

    def add_result(self, record: Dict[str, Any]) -> None:
        """Adds a standardized experiment record."""
        self.results.append(record)

    def get_ranked_models(self) -> List[Dict[str, Any]]:
        """Ranks all models based on primary metric."""
        if not self.results:
            return []

        # Higher is better for accuracy, f1, precision, recall, roc_auc, r2
        # Lower is better for mae, rmse
        reverse = self.primary_metric not in ["mae", "rmse"]

        return sorted(
            self.results,
            key=lambda x: x["metrics"].get(self.primary_metric, 0.0),
            reverse=reverse
        )

    def get_best_model(self) -> Dict[str, Any]:
        """Returns the top performing model record."""
        ranked = self.get_ranked_models()
        if not ranked:
            raise ValueError("No model results recorded.")
        return ranked[0]
