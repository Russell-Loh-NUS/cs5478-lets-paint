"""Submission entrypoint for train_stage3_goal_dagger_v4. Run from the project root."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
runpy.run_path(str(root / "train_stage3_goal_dagger_v4.py"), run_name="__main__")
