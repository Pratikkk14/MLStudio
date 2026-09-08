from typing import Any, Dict, List

class ModelComparator:
    """Aggregates and compares experimental scores of candidate models."""
    def __init__(self, primary_metric: str):
        self.primary_metric = primary_metric
        self.results: List[Dict[str, Any]] = []

    def add_result(self, model_name: str, metrics: Dict[str, float], runtime: float) -> None:
        """Records an experiment result."""
        self.results.append({
            "model_name": model_name,
            "metrics": metrics,
            "runtime_seconds": runtime
        })

    def get_best_model(self) -> Dict[str, Any]:
        """Finds best model based on the primary metric."""
        if not self.results:
            raise ValueError("No results available for comparison.")
        
        # Determine sorting direction (higher is better for f1/acc, lower is better for rmse/mae)
        reverse = True
        if self.primary_metric.lower() in ["mae", "rmse"]:
            reverse = False

        sorted_results = sorted(
            self.results,
            key=lambda x: x["metrics"].get(self.primary_metric.lower(), 0.0),
            reverse=reverse
        )
        return sorted_results[0]
