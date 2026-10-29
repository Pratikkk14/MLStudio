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

def test_loader_directory_single_dataset(tmp_path):
    dir_path = tmp_path / "single_ds"
    dir_path.mkdir()
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(dir_path / "dataset.csv", index=False)
    
    loader = DatasetLoader(dir_path)
    info = loader.get_info()
    assert info["is_directory"] is True
    assert info["is_presplit"] is False
    
    loaded_df = loader.load()
    assert len(loaded_df) == 3

def test_loader_directory_presplit_dataset(tmp_path):
    dir_path = tmp_path / "split_ds"
    dir_path.mkdir()
    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test_df = pd.DataFrame({"a": [5, 6], "target": [1, 0]})
    train_df.to_csv(dir_path / "train.csv", index=False)
    test_df.to_csv(dir_path / "test.csv", index=False)
    
    loader = DatasetLoader(dir_path)
    info = loader.get_info()
    assert info["is_directory"] is True
    assert info["is_presplit"] is True
    assert "train.csv" in info["train_file"]
    assert "test.csv" in info["test_file"]
    
    tr, te = loader.load_splits()
    assert len(tr) == 4
    assert len(te) == 2
