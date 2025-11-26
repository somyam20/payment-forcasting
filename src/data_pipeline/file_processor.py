import os
import uuid
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# config values optional override via env
TEMP_DIR = os.getenv("TEMP_DIR", "data/temp")
SMALL_FILE_ROWS = int(os.getenv("SMALL_FILE_ROWS", 2000))

os.makedirs(TEMP_DIR, exist_ok=True)

def save_stream_to_temp(file_stream, filename: str = None) -> str:
    """
    Save an uploaded file-like object to the temp directory and return path.
    file_stream must support .read().
    """
    filename = filename or f"{uuid.uuid4().hex}"
    local_path = os.path.join(TEMP_DIR, filename)
    with open(local_path, "wb") as f:
        f.write(file_stream.read())
    logger.info(f"Saved upload to {local_path}")
    return local_path

def load_dataframe(local_path: str, sheet_name: str = 0) -> pd.DataFrame:
    """
    Load CSV or Excel into a pandas DataFrame.
    Raises ValueError for unsupported extensions.
    """
    if local_path.lower().endswith(".csv"):
        df = pd.read_csv(local_path)
    elif local_path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(local_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file type. Only CSV and Excel are supported.")
    logger.info(f"Loaded dataframe shape={df.shape} from {local_path}")
    return df

def classify(df) -> str:
    """
    Return 'small' if rows <= SMALL_FILE_ROWS else 'large'.
    """
    n = len(df)
    label = "small" if n <= SMALL_FILE_ROWS else "large"
    logger.info(f"Classified file as '{label}' (rows={n})")
    return label

def sample_rows(df, n: int = 50):
    """
    Return up to n rows as list-of-dicts for LLM context.
    """
    return df.head(n).to_dict(orient="records")
