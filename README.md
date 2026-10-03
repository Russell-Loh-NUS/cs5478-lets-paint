# Lets Paint!

## Environment
- Environment consist of walls at 90 degrees to each other
- Obstacles such as windows and doors are defined as 'holes' in the wall and should not be painted

# Painting
- We simulate painting by having an overlay on each wall surface.
- When the paint roller collides with the wall, we draw a mask on this surface based on the u,v (2D) coordinates of the surface.
- This mask will be used to create a texture which we will reload onto the overlay to simulate the painted wall.
_Note: We have to store the texture locally before loading it again to allow our perception camera to render the updated texture_

# Notes
- Considered simulating painted wall by breaking wall into segments and consider each segment painted once roller touches it. However, this is not feasible as many segments would have to be created if we want it to be more 'granular/detailed'. Moreover, at corners, painting a single side would also result in the other side getting painted (due to a single texture update to the whole object).
- Another alternative considered was to initialize and 'stretch' a plane by noting its start and end collision on the wall surface. However, for each stroke we will have to initialize a plane and overlapping strokes would cause the planes to fight at the same index causing it to not render properly.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pillow pybullet
```

## Run
```bash
.venv/bin/python demo.py # Headless (no GUI)
.venv/bin/python demo.py --gui # Opens the GUI
```