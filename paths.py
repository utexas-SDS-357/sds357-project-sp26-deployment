from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = REPO_ROOT / "data" / "raw"
INT_DATA_PATH = REPO_ROOT / "data" / "intermediate"
FINAL_DATA_PATH = REPO_ROOT / "data" / "final"