import pandas as pd
from src.data.validator import DatasetValidator

def test_validator_empty_dataframe():
    df = pd.DataFrame()
    validator = DatasetValidator(df)
    errors, warnings = validator.validate()
    assert "Dataset is empty." in errors

def test_validator_valid_dataframe():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    validator = DatasetValidator(df)
    errors, warnings = validator.validate()
    assert len(errors) == 0

def test_validator_missing_target():
    df = pd.DataFrame({"col1": [1, 2, 3]})
    validator = DatasetValidator(df, target_column="target")
    errors, warnings = validator.validate()
    assert any("target" in err for err in errors)

def test_validator_warnings():
    df = pd.DataFrame({
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 90% majority (imbalance)
        "const_col": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], # Constant column
        "missing_col": [1, 2, None, 4, 5, 6, 7, 8, 9, 10], # 1 missing value
        "empty_col": [None] * 10, # Completely empty column
        "id_col": [f"ID_{i}" for i in range(10)] # High cardinality / Identifier
    })
    
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    validator = DatasetValidator(df, target_column="target")
    errors, warnings = validator.validate()
    
    # Print warnings to inspect during test run if needed
    print(warnings)
    
    # Verify warnings are triggered
    assert len(errors) == 0
    assert any("imbalance" in w for w in warnings)
    assert any("duplicate" in w for w in warnings)
    assert any("constant" in w for w in warnings)
    assert any("missing value" in w for w in warnings)
    assert any("entirely empty" in w for w in warnings)
    assert any("cardinality" in w for w in warnings)
