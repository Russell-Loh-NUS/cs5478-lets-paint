import argparse
import math
import os

import numpy as np
import pybullet as p
from PIL import Image

from environment import PaintEnvironment

SPEED = 2 # Change this if you want the painting to be done faster

def run(env, waypoints, speed=SPEED, on_tick=None):
    pos = np.array(env.roller.pose()[0])
    for goal, yaw in waypoints:
        goal = np.asarray(goal, float)
        n = max(1, int(np.linalg.norm(goal - pos) / speed / env.dt))
        for i in range(1, n + 1):
            env.roller.set_target(pos + (goal - pos) * i / n, yaw)
            env.step()
            if on_tick:
                on_tick()
        pos = goal

def wall_strokes(wall, r, u_positions, v_lo, v_hi, press=0.012, standoff=0.12):
    yaw = math.atan2(-wall.normal[1], -wall.normal[0])
    near = r - press # Roller centre distance from the plane while pressing
    pts = []
    for u in u_positions:
        for d, v in ((standoff, v_lo), (near, v_lo), (near, v_hi), (standoff, v_hi)):
            pts.append((wall.uv_to_world(u, v) + wall.normal * d, yaw))
    return pts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default="output")
    ap.add_argument("--frame-every", type=int, default=240, help="camera frame period (steps)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = PaintEnvironment(gui=args.gui)
    walls = {w.name: w for w in env.walls}
    r = env.roller.cfg.radius
    frames = []

    def tick():
        if env._steps % args.frame_every == 0:
            frames.append(env.camera.capture()["rgb"])

    path = wall_strokes(walls["south"], r, [0.6, 0.78, 0.96, 1.14, 1.5, 1.68], 0.25, 2.1)
    path += wall_strokes(walls["east"], r, [2.75], 0.25, 2.1)
    try:
        run(env, path, on_tick=tick)
        for _ in range(60):
            env.step()
        env.flush_textures()
    except p.error as e:
        raise SystemExit(f"Stopped early: {e}")

    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"{args.out}/cam_{i:04d}.png")
    for w in env.walls:
        Image.fromarray(w.canvas.texture()).save(f"{args.out}/canvas_{w.name}.png")
    per_wall, total = env.painted_area()
    paintable = sum(w.canvas.paintable_area for w in env.walls)
    for name, area in per_wall.items():
        print(f"  {name:6s} {area:6.2f} m^2")
    print(f"Total painted area: {total:.2f} m^2 of {paintable:.2f} m^2 paintable wall "
          f"({100 * total / paintable:.1f}%)")
    print(f"saved {len(frames)} camera frames and {len(env.walls)} canvases to {args.out}/")
    if args.gui:
        print("Painting Done.")
        try:
            while True:
                env.step()
        except (KeyboardInterrupt, p.error):
            pass
    env.close()
