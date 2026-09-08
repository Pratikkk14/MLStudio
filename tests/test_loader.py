import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DatasetLoader

def test_loader_invalid_file():
    loader = DatasetLoader("non_existent_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()

def test_loader_unsupported_format(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("dummy content")
    loader = DatasetLoader(file_path)
    with pytest.raises(ValueError):
        loader.load()
