# Stage 1: Cartesian Painting DAgger

[train.py](./train.py) applies DAgger to the Stage 1 Cartesian wall-painting
policy. It starts from a behavior-cloned SAC checkpoint and its demonstration
dataset, then collects expert labels at states visited by the learner. Each
rollout mixes expert and deterministic policy actions; the expert-action
fraction decays between rounds. The actor is trained with behavior-cloning
loss, and evaluation applies only the learned policy.

This is an actor-imitation step, not SAC reinforcement-learning fine-tuning:
the SAC critics are not trained. A passing checkpoint should not be treated as
a fine-tuned SAC agent.

## Inputs and defaults

- Initial actor: `runs/sac_demo_v1/bc_initial_sac.zip`
- Initial demonstrations: `runs/sac_demo_v1/demonstrations.npz`
- Output directory: `runs/sac_dagger_v1`
- Evaluation success gate: 75% of evaluation episodes by default
- Walls: south, east, north, and west by default

The actor and demonstration file must match the Stage 1 environment's
observation and action spaces.

## Run

Run from the `sac_project/` directory so the `runs/` paths resolve as shown.
After restoring the support modules listed below:

```bash
python -m dagger.stage_1.train \
  --model runs/sac_demo_v1/bc_initial_sac.zip \
  --initial-data runs/sac_demo_v1/demonstrations.npz \
  --out runs/stage1_dagger
```

Use `python -m dagger.stage_1.train --help` for the training, sampling, and
evaluation options.

## Outputs

- `round_<n>_actor.zip`: actor checkpoint saved after each training round
- `aggregated_demonstrations.npz`: capped dataset of observations and expert
  action labels
- `dagger_results.json`: per-round rollouts and student-only evaluation

The trainer does not create a `best_actor.zip`; use the checkpoint from the
round that meets the success gate, or compare the recorded evaluations.

## Current checkout limitation

The current checkout is missing `train_sac_demonstrations.py`,
`train_sac_windows.py`, and `train_td3_wall.py`, which this trainer imports
directly or through `train_sac_curriculum.py`. Restore those support modules
before running the commands above; even `--help` imports them.
