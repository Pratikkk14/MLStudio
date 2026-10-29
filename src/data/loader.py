from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import pandas as pd

class DatasetLoader:
    """Handles loading tabular data from disk using pandas, supporting both

    single files and directory-based dataset packages (with automatic
    train/test detection).
    """
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.is_presplit: bool = False
        self.train_path: Optional[Path] = None
        self.test_path: Optional[Path] = None
        self.primary_file: Optional[Path] = None
        self._inspect_path()

    def _inspect_path(self):
        if not self.path.exists():
            return

        if self.path.is_file():
            self.primary_file = self.path
            self.is_presplit = False
            return

        # It's a directory: scan for tabular files
        tabular_files: List[Path] = []
        for p in self.path.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"] and not p.name.startswith("."):
                tabular_files.append(p)

        if not tabular_files:
            raise FileNotFoundError(f"No CSV or Excel dataset files found inside directory: {self.path}")

        # Check for train / test split
        train_candidates = [p for p in tabular_files if "train" in p.stem.lower()]
        test_candidates = [p for p in tabular_files if "test" in p.stem.lower() or "val" in p.stem.lower()]

        if train_candidates and test_candidates:
            self.is_presplit = True
            self.train_path = train_candidates[0]
            self.test_path = test_candidates[0]
            self.primary_file = self.train_path
        else:
            self.is_presplit = False
            # Pick the largest tabular file as primary
            tabular_files.sort(key=lambda p: p.stat().st_size, reverse=True)
            self.primary_file = tabular_files[0]

    def get_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the dataset structure."""
        return {
            "path": str(self.path),
            "is_directory": self.path.is_dir(),
            "is_presplit": self.is_presplit,
            "train_file": str(self.train_path) if self.train_path else None,
            "test_file": str(self.test_path) if self.test_path else None,
            "primary_file": str(self.primary_file) if self.primary_file else None,
            "split_strategy": "Pre-split Directory (Dedicated Train/Test)" if self.is_presplit else "Single Dataset (Stratified 80/20 Auto-Split)"
        }

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        elif suffix in [".xls", ".xlsx"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def load(self) -> pd.DataFrame:
        """Loads dataset.

        If a train/test split directory is detected, returns concatenated
        DataFrame with an internal partition attribute.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File or directory not found: {self.path}")

        if self.path.is_file():
            return self._read_file(self.path)

        if self.is_presplit and self.train_path and self.test_path:
            train_df = self._read_file(self.train_path)
            test_df = self._read_file(self.test_path)
            # Mark split indices in dataframe attrs
            combined = pd.concat([train_df, test_df], ignore_index=True)
            combined.attrs["is_presplit"] = True
            combined.attrs["train_len"] = len(train_df)
            combined.attrs["test_len"] = len(test_df)
            return combined
        elif self.primary_file:
            return self._read_file(self.primary_file)
        else:
            raise FileNotFoundError(f"No valid dataset found in: {self.path}")

    def load_splits(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Returns (train_df, test_df).

        If not pre-split, test_df is None.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        if self.is_presplit and self.train_path and self.test_path:
            train_df = self._read_file(self.train_path)
            test_df = self._read_file(self.test_path)
            return train_df, test_df
        else:
            df = self.load()
            return df, None
