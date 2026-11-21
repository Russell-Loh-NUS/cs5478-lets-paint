# Stage 3: Full-Wall DAgger (V4)

[train.py](./train.py) extends the goal-conditioned controller to full-wall
painting with the V8 deterministic coverage planner. The planner remains
necessary at deployment; the neural actor learns low-level motion commands,
not an independent coverage route.

The Stage 3 observation has 56 state values. By default, the trainer
warm-starts the actor from a 48-state Stage 2 checkpoint, preserving its
existing feature weights and initializing the eight new transit/phase inputs
to zero. A compatible 56-state actor can instead be supplied with
`--init-actor`. A fresh Stage 3 DAgger dataset is collected either way.

Each round collects expert/mixed trajectories plus separate student-driven
probes. Labels are retained by motion phase, including cleanup, recovery, and
rare transitions, and the actor is trained with phase-focused weighted
behavior cloning. Evaluation applies only student actions. SAC critics,
replay data, and optimizer state are not trained or transferred.

## Run

Run from `sac_project/`. The Stage 3 default input is
`runs/stage2_recovery_dagger_v3/best_actor.zip`, which differs from Stage 2's
default output directory. If using the Stage 2 default, pass its checkpoint
explicitly and choose a new output directory:

```bash
python -m dagger.stage_3.train \
  --stage2-actor runs/stage2_goal_dagger_v2/best_actor.zip \
  --out runs/stage3_dagger_v4_new
```

The default mode is training. The default stopping gate is a 75% student
success rate, with up to 10 rounds. To resume an existing run, use
`--resume`; the output directory must contain `dagger_results.json`,
`aggregated_demonstrations.npz`, and `best_actor.zip`.

Evaluate a trained actor independently with:

```bash
python -m dagger.stage_3.train \
  --mode evaluate \
  --model runs/stage3_dagger_v4_new/best_actor.zip \
  --out runs/stage3_evaluation
```

Evaluation requires a compatible 56-state Stage 3 checkpoint and writes
`independent_evaluation.json`.

## Outputs

- `warm_started_actor.zip`: initial Stage 3 actor
- `round_<n>_actor.zip`: actor after each DAgger round
- `best_actor.zip`: best evaluated actor, preserved across rounds
- `aggregated_demonstrations.npz`: accumulated phase-labelled data
- `baseline_evaluation.json`, `best_evaluation.json`, and
  `dagger_results.json`: evaluation and training history

## Current checkout limitation

The current checkout is missing `validate_stage3_bottom.py`,
`stage3_bottom_expert.py`, and `train_td3_wall.py`, which provide the
environment and expert dependencies expected by this trainer. Restore those
support modules before running the commands above; even `--help` imports them.
