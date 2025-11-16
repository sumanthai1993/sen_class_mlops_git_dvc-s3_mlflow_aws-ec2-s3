from pathlib import Path
from src.config import (
    BASE_DIR
)

RUN_ID_FILE = BASE_DIR/"mlfow_run_id.txt"

def get_run_id():
    return RUN_ID_FILE.read_text().strip()