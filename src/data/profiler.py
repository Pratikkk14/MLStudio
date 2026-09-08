from typing import Any, Dict, List
import pandas as pd
import numpy as np

class DatasetProfiler:
    """Computes exploratory data analysis metrics, column stats, missing values, and outliers."""
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_summary(self) -> Dict[str, Any]:
        """Calculates dataset-level metrics (shape, types, memory usage, duplicates)."""
        rows, cols = self.data.shape
        numerical_features = list(self.data.select_dtypes(include=[np.number]).columns)
        categorical_features = list(self.data.select_dtypes(exclude=[np.number]).columns)
        
        return {
            "rows": rows,
            "columns": cols,
            "numerical_features_count": len(numerical_features),
            "categorical_features_count": len(categorical_features),
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "memory_usage_bytes": int(self.data.memory_usage(deep=True).sum()),
            "duplicate_rows": int(self.data.duplicated().sum())
        }

    def detect_outliers_iqr(self, col: str) -> Dict[str, Any]:
        """Detects outliers using the Interquartile Range (IQR) method.

        Returns:
            Dict: Outlier metrics including count, percentage, and bounds.
        """
        col_series = self.data[col].dropna()
        if col_series.empty:
            return {"count": 0, "percentage": 0.0, "lower_bound": None, "upper_bound": None}

        q1 = col_series.quantile(0.25)
        q3 = col_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = col_series[(col_series < lower_bound) | (col_series > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = float(outlier_count / len(col_series)) if len(col_series) > 0 else 0.0

        return {
            "count": outlier_count,
            "percentage": outlier_pct,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }

    def profile_columns(self) -> Dict[str, Dict[str, Any]]:
        """Generates statistical profiles for each column in the dataset."""
        profiles = {}
        row_count = len(self.data)

        for col in self.data.columns:
            col_series = self.data[col]
            missing_count = int(col_series.isnull().sum())
            missing_percent = float(missing_count / row_count) if row_count > 0 else 0.0
            unique_count = int(col_series.nunique(dropna=True))

            profile = {
                "dtype": str(col_series.dtype),
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "unique_count": unique_count,
                "cardinality_ratio": float(unique_count / row_count) if row_count > 0 else 0.0
            }

            # Numerical statistics
            if pd.api.types.is_numeric_dtype(col_series.dtype):

                non_null_series = col_series.dropna()
                if not non_null_series.empty:
                    profile.update({
                        "mean": float(non_null_series.mean()),
                        "median": float(non_null_series.median()),
                        "std": float(non_null_series.std()) if len(non_null_series) > 1 else 0.0,
                        "min": float(non_null_series.min()),
                        "max": float(non_null_series.max()),
                        "quantiles": {
                            "25": float(non_null_series.quantile(0.25)),
                            "50": float(non_null_series.quantile(0.50)),
                            "75": float(non_null_series.quantile(0.75))
                        }
                    })
                    # Outliers metrics
                    outliers_info = self.detect_outliers_iqr(col)
                    profile["outliers"] = outliers_info
                else:
                    profile.update({
                        "mean": None, "median": None, "std": None, "min": None, "max": None, "quantiles": None,
                        "outliers": {"count": 0, "percentage": 0.0, "lower_bound": None, "upper_bound": None}
                    })
            # Categorical statistics
            else:
                # Top frequencies mapping
                value_counts = col_series.value_counts(dropna=True).head(5)
                freq_dist = {str(k): int(v) for k, v in value_counts.items()}
                profile["frequency_distribution"] = freq_dist

            profiles[col] = profile

        return profiles
