from typing import Any, Dict, List
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

class ValidationEngine:
    """Manages train/test splits and cross-validation strategies."""
    def __init__(self, task_type: str, test_size: float = 0.2, n_splits: int = 5, random_seed: int = 42):
        self.task_type = task_type
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_seed = random_seed

    def train_test_split(self, X: Any, y: Any) -> List[Any]:
        """Performs initial train/test partitioning."""
        stratify = y if "classification" in self.task_type.lower() else None
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=stratify
        )

    def get_cv_splitter(self) -> Any:
        """Returns appropriate scikit-learn cross-validation splitter."""
        if "classification" in self.task_type.lower():
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        else:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
