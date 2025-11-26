import os
import re
import uuid

ALLOWED_EXTENSIONS = (".csv", ".xls", ".xlsx")

def secure_filename(filename: str) -> str:
    """
    Remove unsafe characters from the filename.
    """
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return f"{uuid.uuid4().hex}_{filename}"

def allowed_file(filename: str) -> bool:
    return filename.lower().endswith(ALLOWED_EXTENSIONS)

def ensure_dir(path: str):
    """
    Creates directory if missing.
    """
    os.makedirs(path, exist_ok=True)
