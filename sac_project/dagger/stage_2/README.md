# Stage 2: Goal-Conditioned DAgger

[train.py](./train.py) trains a goal-conditioned controller for wall painting.
It adds eight planner-goal and phase features to the observation and uses the
`Stage2PatchSweep` expert to label learner-visited states. The deterministic
coverage planner still chooses the route: the learned policy imitates
low-level Cartesian movement rather than learning autonomous route planning.

The trainer creates a new actor; it does not initialize from the Stage 1
checkpoint. Its larger observation space is incompatible with Stage 1
checkpoints and datasets. As in Stage 1, training clones the actor only; SAC
critics are not trained.

## Modes

- `validate-expert`: run the expert and save `expert_validation.json`.
  Training requires this file in the same output directory, and validation
  must report success for every episode.
- `train`: collect DAgger data, clone the actor, and evaluate after each round.
- `evaluate`: run a saved Stage 2 actor without expert actions.

## Run

Run from `sac_project/`. Use the same `--out` path for expert validation and
training:

```bash
python -m dagger.stage_2.train \
  --mode validate-expert \
  --out runs/stage2_goal_dagger_v2

python -m dagger.stage_2.train \
  --mode train \
  --out runs/stage2_goal_dagger_v2
```

Evaluate the selected checkpoint on separate episodes with:

```bash
python -m dagger.stage_2.train \
  --mode evaluate \
  --model runs/stage2_goal_dagger_v2/best_actor.zip \
  --out runs/stage2_evaluation
```

The default success gate is 75%. The default training output is
`runs/stage2_goal_dagger_v2`; the default evaluation count is 16 episodes.

## Outputs

- `expert_validation.json`: prerequisite expert rollout results
- `round_<n>_actor.zip`: actor checkpoint for each round
- `best_actor.zip` and `best_evaluation.json`: best student checkpoint and its
  evaluation
- `aggregated_demonstrations.npz`: accumulated DAgger labels
- `dagger_results.json`: per-round results
- `evaluation.json`: output from `--mode evaluate`

## Current checkout limitation

The current checkout is missing `stage2_expert_recovery.py`,
`train_td3_wall.py`, `train_sac_demonstrations.py`, and
`train_sac_windows.py`. These are imported directly or through the Stage 1
environment and DAgger helpers. Restore them before running this trainer; even
`--help` imports them.
