from __future__ import annotations

import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p

from simulation.paint_environment import PaintEnvironment, PaintConfig
from simulation.room_config import RoomConfig
from simulation.paint_wall import Opening
from simulation.robot import Robot
from evaluation.path_length import PathLengthTracker


class RLCoverageEnv(gym.Env):
    """Gym environment wrapper around the paint environment for RL-driven coverage.

    Notes:
    - Actions are intentionally left as a placeholder (`Discrete(1)`) until you define
      the RL action set. The environment implements reward computation and paint/dip
      semantics so you can plug any action shape later.
    - Each dip replenishes `paint_remaining` to `dip_paint_meters` (default 10.0 m).
    - If the agent attempts to paint beyond the available paint, the environment
      returns an immediate large penalty (-100) and `terminated=True`.
    """

    metadata = {"render.modes": []}

    def __init__(self, *, gui: bool = False, seed: Optional[int] = None,
                 paint_config: Optional[PaintConfig] = None,
                 coverage_target: float = 0.99,
                 weights: Optional[dict] = None,
                 dip_paint_meters: float = 10.0,
                 distance_ref: float = 1.0,
                 max_steps: int = 1000,
                 collision_termination: int = 1,
                 stage: int = 1,
                 wall_name: str = "south"):
        super().__init__()
        self.gui = gui
        self.seed = seed
        self.paint_config = paint_config or PaintConfig()
        self.env = PaintEnvironment(gui=gui, paint=self.paint_config, seed=seed)

        # create robot and tracker; agent/loop may move the robot via planner/controller
        start_xy = ((0.0 + 1.0) / 2.0, (0.0 + 1.0) / 2.0)  # placeholder until reset sets room
        self.robot = Robot(start_xy=start_xy, max_lift=2.5, x_range=(0, 1), y_range=(0, 1))
        self.path_tracker = PathLengthTracker()

        # Action space: continuous target (x, y in world coords) + dip signal in [0,1].
        # The bounds are updated in `reset()` once the room dimensions are known.
        self.action_space = gym.spaces.Box(low=np.array([-10.0, -10.0, 0.0], dtype=np.float32),
                           high=np.array([10.0, 10.0, 1.0], dtype=np.float32), dtype=np.float32)
        # observations: coverage fraction, paint remaining (m), total paintable area
        # Use 1-d boxes so SB3's MultiInputPolicy feature extractors can flatten them.
        self.observation_space = gym.spaces.Dict({
            "coverage": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "paint_remaining": gym.spaces.Box(0.0, np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
            "paintable_area": gym.spaces.Box(0.0, np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
        })

        # default weights from the design screenshot; user can override
        self.weights = weights or {
            "w_c": 100.0,  # coverage reward weight
            "w_d": 1.0,    # distance penalty weight
            "w_o": 100.0,  # collision penalty per collision
            "w_p": 1.0,    # dip penalty weight (tunable)
            "R_f": 50.0,   # completion bonus
        }

        self.coverage_target = coverage_target
        self.dip_paint_meters = dip_paint_meters
        self.distance_ref = distance_ref
        self.max_steps = max_steps
        # number of collision occurrences that trigger terminal failure
        self.collision_termination = collision_termination
        # training stage controls room openings and dip enforcement
        self.stage = int(stage)
        # which wall the RL agent trains on (only this wall's coverage counts)
        self.wall_name = wall_name

        # internal state
        self.paint_remaining = self.dip_paint_meters
        self._last_painted_total = 0.0
        self._last_total_length = 0.0
        self._done = False
        self._step_count = 0
        self._collision_count = 0

        # If stage implies no dip constraint for early stages, set large budget until stage 4
        if self.stage < 4:
            # effectively disable paint exhaustion earlier in curriculum
            self.dip_paint_meters = 1e9

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Rebuild environment/room with provided seed or the env seed
        if seed is None:
            seed = self.seed
        # recreate the paint environment to ensure clean state
        try:
            self.env.close()
        except Exception:
            pass
        # Build a RoomConfig according to the curriculum stage if requested
        room = None
        if self.stage == 1:
            # plain room, no openings
            room = RoomConfig(openings={})
        elif self.stage == 2:
            # single opening (window) on the south wall
            w = RoomConfig()
            win = Opening(u=max(0.5, w.width * 0.3), v=0.8, width=0.9, height=1.0)
            room = RoomConfig(width=w.width, depth=w.depth, height=w.height, openings={"south": (win,)})
        elif self.stage == 3:
            # both a door and a window: door on east, window on south
            w = RoomConfig()
            win = Opening(u=max(0.5, w.width * 0.3), v=0.8, width=0.9, height=1.0)
            door = Opening(u=max(0.3, w.depth * 0.1), v=0.0, width=0.9, height=2.05)
            room = RoomConfig(width=w.width, depth=w.depth, height=w.height,
                              openings={"south": (win,), "east": (door,)})
        else:
            # stage >=4: keep stage 3 layout but enforce dip budget (handled by dip_paint_meters)
            w = RoomConfig()
            win = Opening(u=max(0.5, w.width * 0.3), v=0.8, width=0.9, height=1.0)
            door = Opening(u=max(0.3, w.depth * 0.1), v=0.0, width=0.9, height=2.05)
            room = RoomConfig(width=w.width, depth=w.depth, height=w.height,
                              openings={"south": (win,), "east": (door,)})

        self.env = PaintEnvironment(gui=self.gui, paint=self.paint_config, seed=seed, room=room)
        # reset robot and tracker to sensible defaults based on room
        x_range, y_range = (0.0, 0.0), (0.0, 0.0)
        try:
            x_range, y_range = self._room_xy_ranges()
        except Exception:
            pass
        start_xy = ((x_range[0] + x_range[1]) / 2.0, (y_range[0] + y_range[1]) / 2.0)
        self.robot = Robot(start_xy=start_xy, max_lift=max(w.height for w in self.env.walls),
                           x_range=(x_range[0], x_range[1]), y_range=(y_range[0], y_range[1]))
        self.path_tracker = PathLengthTracker()

        # Update action space bounds to the actual room x/y ranges so the agent outputs
        # valid world coordinates. Use z=0 placeholder and dip in [0,1].
        try:
            (x0, x1), (y0, y1) = self._room_xy_ranges()
            low = np.array([x0 - 0.5, y0 - 0.5, 0.0], dtype=np.float32)
            high = np.array([x1 + 0.5, y1 + 0.5, 1.0], dtype=np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        except Exception:
            pass

        self.paint_remaining = self.dip_paint_meters
        per_wall, total_all = self.env.painted_area()
        # Only focus on the selected wall for training
        total = per_wall.get(self.wall_name, 0.0)
        self._last_painted_total = total
        self._last_total_length = self.path_tracker.get_total_length()
        self._done = False
        self._step_count = 0
        self._collision_count = 0
        obs = self._observation()
        return obs, {}

    def step(self, action):
        if self._done:
            return self._observation(), 0.0, True, False, {}
        self._step_count += 1

        # Placeholder: action currently unused. Advance the physics one step and compute reward.
        # The user will later replace action processing with real motion commands.
        self.env.step()

        # recompute painted area
        per_wall, total_all = self.env.painted_area()
        # compute for selected wall only
        total = per_wall.get(self.wall_name, 0.0)
        # paintable area for the selected wall in m^2
        sel_wall = self.env.walls_by_name.get(self.wall_name)
        if sel_wall is None:
            total_paintable = 0.0
        else:
            total_paintable = float(sel_wall.canvas.paintable.sum()) / (self.env.paint.px_per_m ** 2)

        deltaA = max(0.0, total - self._last_painted_total)
        deltaC = (deltaA / total_paintable) if total_paintable > 0 else 0.0

        # distance travelled since last
        L = self.path_tracker.get_total_length() - self._last_total_length

        # collisions: coarse count of contact points between robot and any wall
        collisions = 0
        try:
            for w in self.env.walls:
                pts = p.getContactPoints(self.robot.id, w.body_id)
                collisions += len(pts)
        except Exception:
            collisions = 0

        # record collision occurrences (count as 1 occurrence even if multiple contact points)
        if collisions > 0:
            self._collision_count += 1

        # dips: no automatic dip handling here; external controller should call `register_dip()`
        unnecessary_dips = 0

        # check paint usage: use path_tracker.distance_since_dip as paint-consumed since last dip
        painted_since_dip = self.path_tracker.distance_since_dip()
        # if the roller has consumed more paint than remaining, mark paint-out
        paint_out = painted_since_dip > self.paint_remaining

        reward = 0.0
        w = self.weights
        reward += w["w_c"] * deltaC
        reward -= w["w_d"] * (L / self.distance_ref)
        reward -= w["w_o"] * collisions
        reward -= w["w_p"] * unnecessary_dips

        # paint out penalty
        terminated = False
        if paint_out:
            reward -= 100.0
            terminated = True

        # terminal collision failure
        if self._collision_count >= self.collision_termination:
            reward -= 100.0
            terminated = True

        # completion bonus (on selected wall)
        coverage_fraction = (total / total_paintable) if total_paintable > 0 else 0.0
        if coverage_fraction >= self.coverage_target:
            reward += w["R_f"]
            terminated = True

        # update bookkeeping
        self._last_painted_total = total
        self._last_total_length = self.path_tracker.get_total_length()

        self._done = terminated
        obs = self._observation()
        return obs, float(reward), terminated, False, {}

    def register_dip(self):
        """Call when the agent performs a dip action: refill paint and reset dip counter."""
        self.paint_remaining = self.dip_paint_meters
        self.path_tracker.reset_dip_counter()

    def _observation(self):
        per_wall, total = self.env.painted_area()
        total_paintable = sum(w.canvas.paintable.sum() for w in self.env.walls) / (self.env.paint.px_per_m ** 2)
        coverage_fraction = (total / total_paintable) if total_paintable > 0 else 0.0
        return {
            "coverage": np.array([float(coverage_fraction)], dtype=np.float32),
            "paint_remaining": np.array([float(self.paint_remaining)], dtype=np.float32),
            "paintable_area": np.array([float(total_paintable)], dtype=np.float32),
        }

    def _room_xy_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xs, ys = [], []
        for w in self.env.walls:
            xs += [float(w.origin[0]), float(w.origin[0] + w.u_dir[0] * w.length)]
            ys += [float(w.origin[1]), float(w.origin[1] + w.u_dir[1] * w.length)]
        return (min(xs), max(xs)), (min(ys), max(ys))

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
