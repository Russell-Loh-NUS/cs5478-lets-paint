# Lets Paint!

A PyBullet-based wall-painting robot simulation. The project combines a coordinate-based room and wall environment, coverage planning, robot motion planning, continuous painting, redipping, replanning, and optional PPO policies for both coverage-target selection and base motion.

## Repository Structure

### Top-level files

- `main.py` — Main entry point. Creates the simulated room, runs the selected coverage strategy, executes routes, handles replanning and redips, and prints/logs run statistics.
- `curriculum_train.py` — Trains the coverage-target PPO policy through four curriculum stages and saves the stage and final `.zip` models.
- `train_motion_rl.py` — Trains or evaluates the optional base-motion PPO policy for trips to the paint can and back to the wall.
- `verify_painting.py` — Physics checks for paint-can solidity, robot/can collisions, IK, wall motion, painting, and retraction behavior. It runs headlessly and may take several minutes.
- `requirements.txt` — Python package dependencies for simulation, RL, and numerical processing.
- `README.md` — Project documentation.
- `.github/` — Repository automation configuration; the `workflows/` subfolder contains GitHub Actions workflows.

### `control/`

- `__init__.py` — Marks the directory as a Python package.
- `coord_robot_controller.py` — Converts planned wall targets into robot actions, checks paint contact, executes approach/retraction, and manages trips to the paint can.
- `robot_controller.py` — Shared robot-control constants and utilities, including the paint/redip distance budget.

### `evaluation/`

- `__init__.py` — Marks the directory as a Python package.
- `coverage.py` — Coverage calculations and coverage-error metrics across wall maps.
- `path_length.py` — Tracks executed robot travel distance and paint consumed since the most recent dip.

### `mapping/`

- `__init__.py` — Marks the directory as a Python package.
- `coord_wall_map.py` — Builds a virtual grid over each continuous wall and determines obstacle, paintable, and painted cells.
- `coverage_map.py` — Stores and evaluates remaining wall coverage and replanning completion status.

### `planning/`

- `__init__.py` — Marks the directory as a Python package.
- `coverage_planner.py` — Implements sweep, optimized PRM, BCD A*, and PPO-guided coverage route construction.
- `motion_planner.py` — Collision-checked classical arm and base motion planning, with Cartesian motion, IK candidate searches, detours, and joint-space recovery.
- `motion_planner_rl.py` — PPO base-motion environment and `BaseNavigator` policy wrapper for paint-can and return trips.
- `rl_env.py` — Gymnasium environment for training the coverage-target PPO policy.

### `simulation/`

- `__init__.py` — Marks the directory as a Python package.
- `paint_environment.py` — Creates the PyBullet world, walls, robot, paint can, camera, and per-step simulation state.
- `paint_canvas.py` — Maintains each wall's paintable and painted pixel masks.
- `paint_wall.py` — Defines wall geometry, openings, wall UV coordinates, and wall-canvas painting operations.
- `paint_can.py` — Defines the paint-can collision geometry and positions used for dipping.
- `room_config.py` — Defines rectangular and L-shaped rooms and the seeded `RoomSampler` used for random layouts.
- `robot.py` — Defines the robot URDF, joint limits, roller geometry, and robot state helpers.
- `pybullet_retry.py` — Utility for handling temporary PyBullet connection failures.

### `models/`

- `ppo_coverage_final.zip` — Default pretrained/final coverage-target PPO model loaded by `--strategy rl`.
- `ppo_stage1.zip` through `ppo_stage4.zip` — Coverage PPO checkpoints from each curriculum stage.
- `motion_rl/` — Base-motion stage checkpoints, final models, and their generated training/evaluation logs.
- `logForFirstRun.txt` and `training_*.log` — Historical or generated training logs.

### `logs/`

Stores timestamped simulation output. The filename records the strategy, RL weight when applicable, seed, rectangular/L-shaped layout, replanning mode, GUI/headless mode, and timestamp. It also records final coverage, runtime, planner time, redips, and replanning comparisons.

### `videos/`

Contains recorded GUI demonstrations for sweep, optimized PRM, BCD A*, coverage RL, and combinations of room layouts and replanning.

> **Visualization note:** the videos are visual demonstrations only. Numerical evaluations and comparisons are run from the headless simulation mode, which disables rendering while retaining the real planning and physics calculations used for measurements.

## Painting Simulation

Each wall has a continuous UV surface backed by a paint canvas. The paintable region excludes doors and windows. A valid paint stroke updates the wall pixels only when the roller makes suitable contact with the wall; coverage is measured from the painted fraction of each wall's paintable pixels.

The planner uses `CoordWallMap` to place wall targets on a virtual grid sized around the robot and roller. The grid is used for planning and replanning, while coverage comes from the actual continuous painted canvas rather than an independent simulated flag.

A route can include a `DipStep` when the paint-distance budget is exhausted or the strategy chooses to return to the paint can. Replanning executes the first pass, measures coverage, and then plans another pass over remaining under-painted cells.

## Room Layout and Randomness

The default room is a fixed rectangular room with dimensions 4 m × 3 m × 2.6 m. `RoomConfig.default()` adds a south window, east door, and north window.

`--seed` uses `RoomSampler` to generate random dimensions, wall openings, opening positions, room height, and optional L-shaped geometry. The seed makes the layout reproducible: the same seed and `--l-shape` settings produce the same room. Seeded rectangular rooms keep consistent seed behavior because rectangular sampling does not consume the L-room draw.

A room is L-shaped when `RoomSampler` chooses an L layout from `--l-shape` probability `p`:

- `--l-shape` (without a value) means probability 1.0: always L-shaped.
- `--l-shape 0.5` means a 50% chance of an L-shaped room.
- `--l-shape 0.0` means no L-shaped rooms.

An L room is formed by cutting a rectangular notch out of one of four corners: southwest, southeast, northeast, or northwest. The notch dimensions and room dimensions are bounded by `RoomSampler` so the remaining L corridor remains reachable and usable. Windows and doors are also sampled within wall boundaries, with margins preventing openings from overlapping or touching wall ends.

The layout can be made deterministic for controlled comparisons with `--seed`. For a fixed seed, `--l-shape` should also be fixed because it changes whether and how the same seed generates an L layout.

An ASCII example of an L-shaped footprint is:

```text
4.0  (6)+---------------+(5)
         |               |
3.0      |               |
         |               +-------+(3)
2.5      |                       |
         |                       |
0.0  (1)+-----------------------+(2)
```

## Coverage Planning Strategies

All coverage strategies work over the same continuous wall surface and use the same roller, dip, and replanning behavior.

### Sweep

A deterministic boustrophedon or back-and-forth sweep follows the wall's continuous UV targets. It is the simplest baseline and has a fixed wall-aligned visitation order. It is useful for predictable coverage and running-time comparisons.

### Optimized PRM

Optimized PRM samples feasible wall points and connects nearby samples into a probabilistic roadmap. It uses nearest-neighbor and 2-opt route improvement, then inserts dip stops with a dynamic-programming pass. The result is generally more locally optimized than the raw sweep but can still depend on the wall representation and sample arrangement.

### BCD A*

BCD A* uses boustrophedon cellular decomposition. It flood-fills connected paintable regions, applies a back-and-forth sweep to each region, and uses grid A* to route between separated regions. A* transitions stay on valid wall cells and detour around openings rather than crossing walls or holes directly.

### Coverage RL

Coverage RL loads a PPO policy and predicts a continuous action `[x, y, dip_score]`:

- `x, y` guide the next wall target.
- `dip_score > 0.5` proposes a dip.
- the planner maps the predicted world target to an unpainted wall grid cell.
- a continuity term prevents the model from repeatedly choosing unrelated distant targets.

The `--weight` value controls this balance:

- `0.25` is the default and keeps stronger emphasis on nearby wall progress.
- Larger values, such as `0.75`, make the model target more influential.
- A value near 1.0 makes the RL target distance more influential while the continuity term remains in the key; this can encourage longer or less local routes.

The coverage RL model trains against coverage gain, distance cost, paint use, collisions, and a completion bonus. It is a planning-level policy, not a low-level arm-controller policy.

### Replanning

By default, the run performs an initial pass and replans up to three times over remaining cells below an 80% coverage threshold. `--disable-replanning` skips replanning and executes only the initial pass.

## Motion Planning Strategies

### Classical Motion Planner

The classical motion planner performs collision-checked Cartesian motion and analytic/greedy IK. It can:

- test candidate base, yaw, shoulder, and extension configurations;
- interpolate and collision-check complete joint trajectories;
- reroute transit through a safe home pose or bounded joint-space search;
- retract from the wall when a direct route is blocked;
- use a fixed or mobile base depending on the motion phase.

### Base-Motion RL

The optional base-motion PPO policy controls only base travel for outbound trips to the paint can and return trips toward the next wall target. It does not drive painting or dipping. The arm remains fixed or is controlled separately by the classical controller. The live policy proposes base positions, and `BaseNavigator` executes the classical controller's collision-checked base motion.

Enable it with:

```bash
python main.py --motion-rl models/motion_rl/<run-name>/motion_rl_final.zip
```

## Setup

Run commands from the repository root: `/Users/huangbin/Desktop/CS5478/Project`.

### Conda environment

```bash
conda create -n lets-paint python=3.11 -y
conda activate lets-paint
python -m pip install -r requirements.txt
```

### pip / venv environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If multiple OpenMP-linked libraries produce a duplicate-library error on macOS, prefix PyBullet and RL commands with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE
```

This project requires PyBullet, NumPy, Pillow, Gymnasium, Stable-Baselines3, and PyTorch. PyBullet may need to be compiled from source on some macOS SDK/Python combinations. Keep the selected Python environment active for both the simulation and training commands.

## Running the Simulation

### Main parameters

- `--strategy STRATEGY`: coverage strategy: `sweep`, `optimized_prm`, `bcd_astar`, or `rl`.
- `--headless`: disable the PyBullet GUI and run numerically/headlessly.
- `--disable-replanning`: execute only the initial pass instead of replanning remaining cells.
- `--seed SEED`: generate a reproducible seeded room. Omitting it uses the fixed default room.
- `--l-shape [PROBABILITY]`: enable L rooms. `--l-shape` is 100%; `--l-shape 0.5` is 50%; omit it for 0% in a seeded-room run.
- `--weight WEIGHT`: coverage-RL continuity weight. Default: `0.25`.
- `--rl-model PATH`: load a coverage PPO model from a custom `.zip` path. Default: `models/ppo_coverage_final.zip`.
- `--motion-rl MODEL`: use a trained base-motion PPO model for paint-can and return trips.
- `--motion-rl-fallback`: fall back to the classical motion planner if a base-motion RL leg fails.
- `--motion-rl-waypoint`: add a safety waypoint beside the paint can before an RL return leg.

Omitting `--strategy` uses the default configured strategy in `main.py`, currently `rl`.

### GUI and headless examples

```bash
# Open the simulation GUI
KMP_DUPLICATE_LIB_OK=TRUE python main.py

# Run headlessly
KMP_DUPLICATE_LIB_OK=TRUE python main.py --headless

# Fixed seed, rectangular layout
KMP_DUPLICATE_LIB_OK=TRUE python main.py --seed 1 --headless

# Fixed seed, always L-shaped
KMP_DUPLICATE_LIB_OK=TRUE python main.py --seed 3 --l-shape --headless

# Seeded room with a 50% L-room probability
KMP_DUPLICATE_LIB_OK=TRUE python main.py --seed 3 --l-shape 0.5 --headless
```

### Coverage strategy examples

```bash
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy sweep --headless
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy optimized_prm --headless
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy bcd_astar --headless
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy rl --weight 0.25 --headless
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy rl --weight 0.75 --headless
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy rl --rl-model models/ppo_coverage_final.zip --headless
```

### Replanning examples

```bash
# Default: initial pass plus up to three replanning attempts
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy rl --headless

# Initial pass only
KMP_DUPLICATE_LIB_OK=TRUE python main.py --strategy rl --disable-replanning --headless
```

### Base-motion RL examples

```bash
KMP_DUPLICATE_LIB_OK=TRUE python main.py --motion-rl models/motion_rl/<run-name>/motion_rl_final.zip --headless

KMP_DUPLICATE_LIB_OK=TRUE python main.py --motion-rl models/motion_rl/<run-name>/motion_rl_final.zip --motion-rl-fallback --motion-rl-waypoint --headless
```

## Training Coverage RL

Coverage RL is trained with Stable-Baselines3 PPO on `RLCoverageEnv`. The four-stage curriculum starts with a plain room, adds a south window, adds door/window layouts, and then continues on the same door/window layout with the dip budget enforced.

### Curriculum training parameters

- `--timesteps`: PPO timesteps per curriculum stage. Default: `2000`.
- `--wall`: wall used for training. Default: `south`.
- `--out`: model/log output directory. Default: `models`.
- `--seed`: environment seed. Default: `0`.
- `--eval`: evaluate the final model after training.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python curriculum_train.py --timesteps 50000 --wall south --out models --seed 0

# Quick smoke training
KMP_DUPLICATE_LIB_OK=TRUE python curriculum_train.py --timesteps 200 --wall south --out /tmp/models_test --seed 0

# Train and evaluate
KMP_DUPLICATE_LIB_OK=TRUE python curriculum_train.py --timesteps 50000 --wall south --out models --seed 0 --eval
```

The default output is `models/ppo_coverage_final.zip`, with stage checkpoints saved as `models/ppo_stage1.zip` through `models/ppo_stage4.zip`.

## Training Base-Motion RL

### Base-motion training parameters

- `--timesteps`: timesteps per curriculum stage. Default: `100000`.
- `--stages`: number of stages, `1`, `2`, or `3`. Default: `3`.
- `--n-envs`: number of parallel environments. Default: `1`; values above 1 use `SubprocVecEnv`.
- `--seed`: training seed. Default: `0`.
- `--out`: output directory. Default: `models/motion_rl`.
- `--phase`: `outbound`, `return`, or `mixed`. Default: `mixed`.
- `--eval-episodes`: number of evaluation episodes after each stage. Default: `50`.
- `--eval-only MODEL`: evaluate an existing model instead of training.
- `--l-prob PROBABILITY`: override the L-room probability during evaluation.
- `--resume MODEL`: resume from an existing model.
- `--first-stage 1|2|3`: first curriculum stage used with `--resume`.
- `--can-between PROBABILITY`: share of return episodes that start with the can between the base and wall.

```bash
# Train the full three-stage curriculum
KMP_DUPLICATE_LIB_OK=TRUE python train_motion_rl.py --timesteps 200000 --out models/motion_rl/my_run --seed 0

# Evaluate an existing model on 50% L-shaped rooms
KMP_DUPLICATE_LIB_OK=TRUE python train_motion_rl.py --eval-only models/motion_rl/my_run/motion_rl_final.zip --l-prob 0.5 --eval-episodes 50

# Resume from stage 2 onward
KMP_DUPLICATE_LIB_OK=TRUE python train_motion_rl.py --resume models/motion_rl/my_run/motion_rl_final.zip --first-stage 2 --stages 3 --timesteps 100000
```

## DAgger Training and Room Runner

DAgger (Dataset Aggregation) improves a policy using an expert controller:
the current policy visits states, the expert labels those states with the
actions it would take, and the resulting state/action examples are added to
the training data. The policy is then behavior-cloned on the growing dataset,
and the process repeats. This helps train on states the policy actually
encounters, not only states from expert demonstrations. The stage trainers
train the actor by imitation; they do not train SAC critics.

The stage-specific trainers and instructions are in
[sac_project/dagger/](sac_project/dagger/). In contrast,
[sac_project/dagger_main.py](sac_project/dagger_main.py) is a room-painting
runner, not a DAgger trainer. It loads an already-trained Stage 3 actor
(56-state observation, 4-action policy), paints the selected walls in one
persistent PyBullet room, and handles paint-can refills. It writes
`summary.json` and `diagnostics.jsonl` under the output directory.

From the repository root, run the actor headlessly on a reproducible room:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python sac_project/dagger_main.py \
  --paint-actor sac_project/runs/stage3_dagger_v4/best_actor.zip \
  --room-seed 42
```


## Physics and Painting Verification

The verification script uses real PyBullet DIRECT physics and no GUI. It checks targets, can solidity, collisions, IK, and painting behavior.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python verify_painting.py
```

It may take several minutes. A passing run should print `PASS` checks and finish without missed-target or collision assertions.
