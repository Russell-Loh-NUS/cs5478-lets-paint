# Robotic Wall Painting — Clean Submission

Hybrid coverage-planning + DAgger robotic painting system (PyBullet).

## Layout
- `dagger/stage_1/`: early DAgger trainer entrypoint.
- `dagger/stage_2/`: goal-conditioned Stage 2 DAgger trainer.
- `dagger/stage_3/`: final Stage 3 V4 painting actor and training entrypoint.
- `dagger/stage_4/`: final Stage 4 V2 paint-dipping actor and training entrypoint.
- `../planning/`: shared coverage and collision-aware motion planning modules.
- `../simulation/`: shared original PyBullet simulation modules.
- `results/`: selected evaluation JSON files.
- `runs/`: final model checkpoints at paths expected by original demo scripts.

The simulation and planning modules are shared from the repository root; the duplicate copies previously under `sac_project/` have been removed. Scripts that need these modules add the repository root to Python's import path. `dagger/stage_*/train.py` are convenient wrappers, not rewritten trainers.

## Installation
From `sac_project/`: `python -m pip install -r requirements.txt`

## Four-wall headless demo
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u four_wall_coverage_demo.py \
  --paint-actor runs/stage3_dagger_v4/best_actor.zip \
  --dip-actor runs/stage4_dip_v2/best_actor.zip \
  --walls south east north west \
  --paint-distance-m 12 --low-fraction 0.15 \
  --max-refills 100 --max-actions 30000 --out runs/submission_demo
```

## Notes
- Stage 3 V4 and Stage 4 V2 are the final trained neural policies; they are not retrained by the demo.
- Refill travel and wall switching use geometric/rule-based control, not learned navigation.
- Four walls are separate environment episodes, not one continuous physical traversal.
- Existing results show incomplete wall coverage and some failures; do not report a 100% success rate.
- Stage 1 and Stage 2 are selected from surviving trainers; no original Stage 1/2 best checkpoints were found in the uploaded archive.
- Training wrappers have not been run to completion; full demo reproduction requires the declared dependencies and PyBullet runtime.
