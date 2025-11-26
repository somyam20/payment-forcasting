import pandas as pd

def validate_target_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

def validate_numeric_column(df: pd.DataFrame, col: str):
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric for LSTM forecasting.")

def validate_dataframe_not_empty(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")
