import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def safe_describe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return df.describe(include='all') as a JSON-serializable dict.
    Coerces non-serializable numpy values to Python native types.
    """
    desc = df.describe(include="all", datetime_is_numeric=True).to_dict()
    # convert numpy types
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    return _convert(desc)

def get_null_counts(df: pd.DataFrame) -> Dict[str, int]:
    return df.isnull().sum().to_dict()

def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    return df.dtypes.astype(str).to_dict()

def get_modes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return first mode for each column when available.
    """
    modes = {}
    for col in df.columns:
        try:
            m = df[col].mode()
            modes[col] = None if m.empty else (m.iloc[0] if len(m) > 0 else None)
        except Exception:
            modes[col] = None
    return modes

def get_descriptive_stats(df) -> Dict[str, Any]:
    """
    Returns a dictionary with:
    - describe (summary stats)
    - null_counts
    - column_types
    - mode
    - basic correlations for numeric columns
    """
    stats = {}
    stats["describe"] = safe_describe(df)
    stats["null_counts"] = get_null_counts(df)
    stats["column_types"] = get_column_types(df)
    stats["mode"] = get_modes(df)

    # correlations: numeric only, return as dict of dict
    try:
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] >= 2:
            corr = numeric.corr().to_dict()
            stats["correlations"] = corr
        else:
            stats["correlations"] = {}
    except Exception as e:
        logger.warning(f"correlation compute failed: {e}")
        stats["correlations"] = {}

    return stats
