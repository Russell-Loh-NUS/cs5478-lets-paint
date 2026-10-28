# Lets Paint!

## Environment
- Environment consist of walls at 90 degrees to each other
- The room layout is a `RoomConfig` (`simulation/room_config.py`). By default it is a fixed 4 x 3 x 2.6 m room. Pass `seed=` (or call `env.reset(seed=...)`) to get a random room from `RoomSampler`; the same seed always gives the same room
- Rooms can also be L shaped: a rectangle with one corner notched out (6 walls, named `south east north west` plus `south_2`/`west_2`/... for the extra ones). `RoomSampler(l_shape_probability=p)` makes a seeded room L shaped with chance p, and `RoomConfig(notch=(corner, notch_width, notch_depth))` builds one by hand
- Obstacles such as windows and doors are defined as 'holes' in the wall and should not be painted

# Painting
- We simulate painting by having an overlay on each wall surface.
- When the paint roller is in contact with the wall and both of its ends are touching the wall (so a tilted roller touching with one end does not paint), we draw a mask on this surface based on the u,v (2D) coordinates of the surface.
- This mask will be used to create a texture which we will reload onto the overlay to simulate the painted wall.
_Note: We have to store the texture locally before loading it again to allow our perception camera to render the updated texture_

# Room Layout
Example of room layout generation - pair of points (x,y) for this example are (1) - (0,0), (2) - (6,0), (3) - (6,2.5), (4) - (4,2.5), (5) - (4,4), (6) - (0,4)

4.0  (6)+---------------+(5)
         |               |
3.5      |               |
         |               |
3.0      |               | Extension of room to create L shaped room, we refer to the part we extend as a notch.
         |               |
2.5      |               +-------+(3)     <- (4) is at the + on the left of 3, notch refers to the extension between the length of (1-6) and (4-5) that creates the L
         |                       |
2.0      |                       |
         |                       |
1.0      |                       |
         |                       |
0.0  (1)+-----------------------+(2)
         x=0     x=2     x=4     x=6

# Notes
- Considered simulating painted wall by breaking wall into segments and consider each segment painted once roller touches it. However, this is not feasible as many segments would have to be created if we want it to be more 'granular/detailed'. Moreover, at corners, painting a single side would also result in the other side getting painted (due to a single texture update to the whole object).
- Another alternative considered was to initialize and 'stretch' a plane by noting its start and end collision on the wall surface. However, for each stroke we will have to initialize a plane and overlapping strokes would cause the planes to fight at the same index causing it to not render properly.
- Considered adding partitions within room to make motion of robot more sophisticated, but there is the challenge of appropriately positioning the partitions such that it can appropriately fit the robot especially with randomisation of seed. Instead, we create L shaped rooms which is determined by the notch (the area of where the extension of the room will be). At least, the shortest path between paint can and robot wont be the straight line distance.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```
_Note: `pybullet` has no prebuilt macOS wheel, so it is compiled from source. This can fail on recent macOS SDKs (zlib's `zutil.h` redefines `fdopen`); building from the sdist after deleting the `#define fdopen(fd, mode) NULL` lines in `examples/ThirdPartyLibs/zlib/zutil.h` works around it._

## Run
Run all commands from the project root.
```bash
.venv/bin/python main.py # Opens the GUI
.venv/bin/python main.py --headless # Full coverage without the GUI
.venv/bin/python main.py --seed 3 # Random room from seed 3 (same seed, same room)
.venv/bin/python main.py --seed 3 --l-shape # Same, but always L shaped (--l-shape 0.5 = 50% chance)
.venv/bin/python verify_painting.py # Checks of the painting setup (DIRECT physics, no GUI, takes a few minutes)
```
In the GUI, use the arrow keys to orbit, shift + arrow keys to pan, `=` / `-` to zoom and the number keys 1-5 for preset views. Closing the window mid-run stops the run early with partial results.

## RL training and strategy options
You can train an RL policy with the provided curriculum trainer and then run the planner using that policy.

### Train the RL policy (curriculum Stage 1 → 4)
Run a full curriculum training (example uses 50k timesteps per stage):

```bash
# macOS OpenMP workaround if needed
KMP_DUPLICATE_LIB_OK=TRUE python3 curriculum_train.py --timesteps 50000 --wall south --out models --seed 0
```

Quick smoke run (short training):

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 curriculum_train.py --timesteps 200 --wall south --out /tmp/models_test
```

The final model is saved as `models/ppo_coverage_final.zip`. Training logs are written to `models/training_<timestamp>.log`.

### Run the simulation with a chosen strategy
Use `--strategy` to pick a coverage strategy. Examples:

```bash
# Use the trained RL policy (planner-driven) in a single pass (no replanning)
KMP_DUPLICATE_LIB_OK=TRUE python3 main.py --strategy rl --disable-replanning

# Use the RL planner but allow replanning (multiple attempts)
python3 main.py --strategy rl --headless

# Run BCD A* strategy
python3 main.py --strategy bcd_astar --headless

# Run PRM strategy
python3 main.py --strategy prm --headless

# Run sweep strategy
python3 main.py --strategy sweep --headless
```

Notes:
- By default `main` looks for the RL model at `models/ppo_coverage_final.zip`. If you prefer, place your model file there before running.
- You can disable replanning entirely with `--disable-replanning` (single planned pass only).
- The RL action contract is continuous `[x, y, dip_score]` where `dip_score > 0.5` triggers a dip; the planner maps `(x,y)` to the nearest wall grid cell when building the route.

