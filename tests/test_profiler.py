import pandas as pd
from src.data.profiler import DatasetProfiler

def test_profiler_summary():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "cat": ["a", "b", "c"]
    })
    profiler = DatasetProfiler(df)
    summary = profiler.get_summary()
    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["numerical_features_count"] == 1
    assert summary["categorical_features_count"] == 1
    assert summary["duplicate_rows"] == 0

def test_profiler_columns():
    df = pd.DataFrame({
        "num": [1.0, 2.0, None],
    })
    profiler = DatasetProfiler(df)
    profiles = profiler.profile_columns()
    assert "num" in profiles
    assert profiles["num"]["missing_count"] == 1
    assert profiles["num"]["unique_count"] == 2
    assert profiles["num"]["mean"] == 1.5

def test_profiler_outliers():
    # Outlier sequence: 10, 12, 11, 13, 14, 15, 12, 10, 100 (outlier)
    # Median is around 12. IQR is Q3(14) - Q1(10.5) = 3.5. Upper bound is 14 + 1.5 * 3.5 = 19.25. 100 exceeds this.
    df = pd.DataFrame({
        "vals": [10, 12, 11, 13, 14, 15, 12, 10, 100]
    })
    profiler = DatasetProfiler(df)
    profiles = profiler.profile_columns()
    
    outliers = profiles["vals"]["outliers"]
    assert outliers["count"] == 1
    assert outliers["upper_bound"] < 100.0
    assert 100.0 > outliers["upper_bound"]
