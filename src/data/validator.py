from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

class DatasetValidator:
    """Performs static validation checks and flags warnings/errors on pandas dataframes."""
    def __init__(self, data: pd.DataFrame, target_column: Optional[str] = None):
        self.data = data
        self.target_column = target_column

    def validate(self) -> Tuple[List[str], List[str]]:
        """Validates the dataset.

        Returns:
            Tuple[List[str], List[str]]: (errors, warnings)
        """
        errors = []
        warnings = []

        # Row check
        if self.data.empty:
            errors.append("Dataset is empty.")
            return errors, warnings

        row_count = len(self.data)
        col_count = len(self.data.columns)

        if row_count < 10:
            warnings.append(f"Dataset is extremely small ({row_count} rows).")

        # Target checks
        if self.target_column:
            if self.target_column not in self.data.columns:
                errors.append(f"Target column '{self.target_column}' not found in dataset columns.")
            else:
                target_series = self.data[self.target_column]
                unique_vals = target_series.dropna().nunique()
                if unique_vals <= 1:
                    errors.append(f"Target column '{self.target_column}' contains only {unique_vals} valid class(es). Cannot perform training.")
                
                # Class imbalance check for classification
                # If target is categorical or integer, check distribution
                if unique_vals == 2 or (unique_vals > 2 and unique_vals <= 20 and not np.issubdtype(target_series.dtype, np.floating)):
                    val_counts = target_series.value_counts(normalize=True)
                    if val_counts.iloc[0] > 0.85:
                        imbalance_pct = val_counts.iloc[0] * 100
                        warnings.append(f"Target class imbalance detected: majority class represents {imbalance_pct:.1f}% of observations.")

        # Duplicate row check
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"{duplicate_count} duplicate rows detected.")

        # Column checks
        for col in self.data.columns:
            if col == self.target_column:
                continue
                
            col_series = self.data[col]
            missing_count = col_series.isnull().sum()
            unique_count = col_series.nunique(dropna=True)
            
            # Completely missing column
            if missing_count == row_count:
                warnings.append(f"Column '{col}' is entirely empty (all values are missing).")
                continue
                
            # Missing value check
            if missing_count > 0:
                missing_pct = (missing_count / row_count) * 100
                warnings.append(f"Column '{col}' has {missing_count} missing values ({missing_pct:.1f}%).")

            # Constant column
            if unique_count == 1:
                warnings.append(f"Column '{col}' is constant (contains only 1 unique value).")

            # Identifier / high cardinality check
            if unique_count > 1 and not pd.api.types.is_numeric_dtype(col_series.dtype):
                uniqueness_ratio = unique_count / row_count

                col_name_lower = col.lower()
                is_id_name = any(kw in col_name_lower for kw in ["id", "key", "uuid", "code", "index"])
                
                if uniqueness_ratio > 0.90 or (is_id_name and unique_count > 10):
                    warnings.append(f"Column '{col}' has high cardinality ({unique_count} unique values). Likely identifier.")

        return errors, warnings
