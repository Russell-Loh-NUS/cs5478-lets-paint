# Stage 4: Opening-Aware Wall-Painting DAgger

[train.py](./train.py) currently implements an opening-aware refinement of the
Stage 3 wall-painting actor. Stage 4 generates safe wall segments around
openings, uses randomized room layouts, and collects expert labels on
student-visited states before behavior-cloning the actor.

## Modes

- `validate-expert`: run and record opening-aware expert episodes.
- `train`: collect DAgger episodes and fit the actor. Expert validation is
  skipped unless `--validate-before-train` is supplied.
- `fit-saved`: fit from episodes already saved under `dagger_episodes/`.
- `evaluate`: evaluate a saved policy with `--policy-mode student`, which
  applies no expert actions.

The trainer requires a compatible 56-state Stage 3 actor. The defaults for
`--stage3-actor` and `--out` are rooted under this `stage_4/` folder. When
using the Stage 3 run directory under `sac_project/runs/`, pass the actor path
explicitly as shown below.

## Run

Run from `sac_project/`. Optionally validate the expert first:

```bash
python -m dagger.stage_4.train \
  --mode validate-expert \
  --out runs/stage4_openings
```

For a fresh training run, set `--start-round 1`. The default is round 2 and
expects `round_1_actor.zip` to already exist in the output directory.

```bash
python -m dagger.stage_4.train \
  --mode train \
  --stage3-actor runs/stage3_dagger_v4/best_actor.zip \
  --out runs/stage4_openings \
  --start-round 1 \
  --rounds 2
```

To run validation during the training invocation, add
`--validate-before-train`. Training uses two rounds by default.

Evaluate the resulting candidate on student-only episodes with:

```bash
python -m dagger.stage_4.train \
  --mode evaluate \
  --policy-mode student \
  --model runs/stage4_openings/candidate_actor.zip \
  --out runs/stage4_evaluation
```

Evaluation writes `student_evaluation.json` and reports the share of episodes
that reach the configured coverage target (80% by default).

## Outputs

- `dagger_episodes/`: saved episode datasets and JSON rollout reports
- `round_<n>_actor.zip`: actor after each fitting round
- `candidate_actor.zip`: final candidate; evaluate it on unseen seeds before
  deployment
- `expert_validation.json`: output from `--mode validate-expert`
- `student_evaluation.json`: output from `--mode evaluate`

The trainer does not select or save a `best_actor.zip`; `candidate_actor.zip`
is not automatically promoted after evaluation.

## Current checkout limitation

The current checkout is missing `train_stage3_goal_dagger_v4.py`,
`stage3_bottom_expert.py`, and `train_td3_wall.py`, which this trainer imports.
Restore those support modules before running the commands above; even
`--help` imports them.
