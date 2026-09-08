from pathlib import Path
from typing import Union
import pandas as pd

class DatasetLoader:
    """Handles loading tabular data from disk using pandas."""
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """Loads dataset from file_path. Supports CSV and Excel."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.file_path)
        elif suffix in [".xls", ".xlsx"]:
            return pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
