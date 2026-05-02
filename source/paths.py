from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in _THIS_FILE.parents if (parent / "data").exists())

RAW_DATA_PATH = REPO_ROOT / "data" / "raw"
INT_DATA_PATH = REPO_ROOT / "data" / "intermediate"
FINAL_DATA_PATH = REPO_ROOT / "data" / "final"