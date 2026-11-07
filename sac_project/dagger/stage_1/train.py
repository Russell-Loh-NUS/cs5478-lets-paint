"""Submission entrypoint for train_sac_dagger. Run from the project root."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
runpy.run_path(str(root / "train_sac_dagger.py"), run_name="__main__")
