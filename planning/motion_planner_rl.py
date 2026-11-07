# RL base-motion planning for the paint-can (redip) trips.
#
# Scope
# -----
# The policy only drives the *base* into a position band - it never touches the arm, never dips and
# never paints.  The trip is made with the arm in a fixed STOWED pose (shoulder vertical, extension
# fully retracted), so the collision footprint is just the chassis and the policy needs no arm state.
#
# OUTBOUND  anywhere        ->  a ring around the can      (arm-only dip takes over)
# RETURN    around the can  ->  a band in front of the next wall target (arm-only approach takes over)
#
# Heading is NOT part of the goal.  With the base fixed, the arm planner derives the yaw itself
# (it points the arm at the target) and turns the stowed arm in place, so only the base POSITION
# matters.  The success test is a distance band, in metres, from the goal point -
#   dip     band is fixed -                 cfg.dip_band
#   paint   band depends on the target height, from the arm's reach (paint_band()).
#
# Two users share the same sensing / observation / action code (_MotionCore) -
#   RLMotionEnv    gym env for training - the base is teleported (no physics stepping), so it is fast.
#   BaseNavigator  runs a trained policy on the live robot - it only proposes the next base pose and
#                  the controller executes it.
#
# Action   Box(2) in [-1, 1] -  [forward/backward, turn].  Forward is along the chassis heading
#          (the arm's +x axis) - the base is moved as a unicycle on top of the robot's x / y / yaw
#          joints.  Yaw is clamped to the joint's +/-pi stops and never wrapped.
# Obs      distance / bearing to the goal point, the SIGNED turn that faces it without crossing the
#          yaw stops, yaw, phase flag, previous action, the goal band, distance to the four base
#          limits and a 360 degree ring of range rays that start at the chassis outline.
# Limits   the base centre is clamped to the room's bounding box inset by 0.6 m, exactly like the
#          real base joints in main.py (BASE_CLEARANCE) - a move past it just stops. The notch walls
#          of an L room are NOT covered by that rectangle, so a soft penalty asks for the same
#          0.6 m centre-to-wall distance from every wall.
# Reward   shaped progress toward the band, time + smoothness cost, soft clearance and 0.6 m inset
#          penalties, big bonus for entering the band, big penalty for a collision or for leaving
#          the room outline (e.g. through a door gap).
#
# Not thread safe - PyBullet module functions use the default client, so run one env per process
# (use SubprocVecEnv for parallel training).
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Optional

import gymnasium as gym
import numpy as np
import pybullet as p

from simulation.paint_can import PaintCan
from simulation.paint_environment import PaintConfig, PaintEnvironment
from simulation.robot import Robot
from simulation.room_config import RoomSampler

CHASSIS_HALF = (0.275, 0.32)  # half extents (x, y) of the chassis box in robot.py
RAY_HEIGHT = 0.12             # rays are cast at mid chassis height
OUTBOUND, RETURN = 0, 1

# Stowed arm used for all base travel - shoulder vertical, extension fully retracted.
STOW_TILT = -math.pi / 2
YAW_LIMIT = 3.14159  # base yaw joint stop in the URDF (slightly under pi) - the robot rejects values past it

# Arm reach geometry (see MotionPlanner) - the roller can be FIXED_REACH + extension away from the
# shoulder, and the shoulder sits SHOULDER_Z above the floor, SHOULDER_OFFSET ahead of the base.
REACH_MIN = 0.10 + 0.55 + 0.21 + Robot.EXTENSION_MIN
REACH_MAX = 0.10 + 0.55 + 0.21 + Robot.EXTENSION_MAX
SHOULDER_Z = 1.0
SHOULDER_OFFSET = 0.02


def stowed_joints(x: float, y: float, yaw: float):
    return [x, y, yaw, STOW_TILT, Robot.EXTENSION_MIN, 0.0, 0.0]


APPROACH_OFFSET = 0.25  # the arm first goes to a point this far in front of the target (RETRACT_DISTANCE)
SWEEP_MIN = 1.15        # below the shoulder the arm swings down through horizontal, where the roller
                        # is at least REACH_MIN out - the wall must be farther than that (+ roller radius)


def paint_band(target_z: float, margin: float = 0.10, floor: float = 0.6):
    # Horizontal base-to-target distances from which the arm alone can bring the roller to a wall
    # target of this height - from the stowed (vertical) arm to the approach point 0.25 m in front of
    # the target, then onto the target. A mid-height cell needs the base roughly 1.3-1.6 m away, a
    # low one 1.15-1.4 m (the swing down), a high one only 0.6-0.9 m. `margin` keeps away from the
    # edges - `floor` is the base inset, since the base can never be closer to the wall than that.
    dz = target_z - SHOULDER_Z
    lo = math.sqrt(max(0.0, REACH_MIN ** 2 - dz ** 2)) + SHOULDER_OFFSET + APPROACH_OFFSET + margin
    hi = math.sqrt(max(0.0, REACH_MAX ** 2 - dz ** 2)) + SHOULDER_OFFSET - margin
    if target_z < SHOULDER_Z:
        lo = max(lo, SWEEP_MIN)
    lo = max(lo, floor)
    return lo, max(hi, lo + 0.05)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def feasible_turn(yaw: float, bearing_abs: float) -> float:
    # Signed turn that faces `bearing_abs` WITHOUT crossing the yaw stops at +/-pi. It equals the
    # wrapped shortest turn unless that one would run through a stop - then it is the long way round.
    turn = wrap_angle(bearing_abs - yaw)
    target = yaw + turn
    if target > YAW_LIMIT:
        return turn - 2.0 * math.pi
    if target < -YAW_LIMIT:
        return turn + 2.0 * math.pi
    return turn


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def distance_to_polygon_edges(x: float, y: float, poly: np.ndarray) -> float:
    best = math.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = float(np.clip(np.dot((x - a[0], y - a[1]), ab) / np.dot(ab, ab), 0.0, 1.0))
        best = min(best, math.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1])))
    return best


@dataclass
class MotionRLConfig:
    # --- scenario -------------------------------------------------------------------------
    phase: str = "mixed"               # "outbound", "return" or "mixed"
    l_shape_probability: float = 0.0   # chance a sampled room is L shaped
    keep_openings: bool = False        # keep doors / windows from the room sampler
    rebuild_room_every: int = 20       # episodes per sampled room (rebuilding walls is slow)
    can_wall_clearance: float = 0.8    # keep the can this far from every wall
    spawn_margin: float = 0.2          # extra free space required around spawned poses
    # Outbound starts look like the base while painting - 0.8-1.6 m in front of a wall, any heading.
    start_wall_range: tuple = (0.8, 1.6)
    start_anywhere_prob: float = 0.3   # otherwise any free pose in the room
    # Return starts are around the can, where a dip leaves the base, with any heading.
    return_start_range: tuple = (0.8, 1.5)
    target_z_range: tuple = (0.15, 2.4)  # heights of the wall targets used for returns
    # --- goal bands (horizontal distance from the goal point, metres) ---------------------------
    # Measured - the arm-only dip works from about 0.8-1.4 m from the can centre at any heading.
    dip_band: tuple = (0.9, 1.3)
    band_margin: float = 0.10          # margin inside the arm's reach for wall targets
    # --- action scaling -------------------------------------------------------------------
    max_forward: float = 0.10          # m per step
    max_reverse: float = 0.05          # reversing is slower
    max_turn: float = 0.25             # rad per step
    # --- sensing / collision ----------------------------------------------------------------
    n_rays: int = 24
    ray_range: float = 3.0
    collision_margin: float = 0.02     # anything closer than this to a wall / the can is a collision
    clearance_soft: float = 0.30       # clearance below which the soft penalty ramps in
    # The real base joints stop `base_inset` short of the room's bounding box (main.py
    # BASE_CLEARANCE) - modelled as a hard clamp. The same distance is also asked of EVERY wall
    # (notch walls included) through a soft penalty, since the joint limits can't enforce that.
    base_inset: float = 0.60
    inset_full_penalty: float = 0.35   # centre-to-wall distance at which the inset penalty is full
    # --- episode --------------------------------------------------------------------------
    max_steps: int = 400
    # --- reward weights -------------------------------------------------------------------
    w_progress: float = 10.0           # per metre closer to the band
    w_time: float = 0.05
    w_action: float = 0.01
    w_clearance: float = 0.25          # chassis-outline clearance (walls and can)
    w_inset: float = 1.0               # centre closer than base_inset to any wall
    r_success: float = 50.0
    r_collision: float = 50.0


class _MotionCore:
    # Sensing, observation, limits and action handling shared by the training env and the live
    # navigator, so the policy sees exactly the same inputs in both. Subclasses provide - cfg,
    # robot, _poly, x, y, yaw, phase, goal_xy, band, _prev_action and _obstacle_bodies().

    def _init_rays(self):
        # Ray geometry in the robot frame - each ray starts on the chassis outline.
        phi = 2.0 * np.pi * np.arange(self.cfg.n_rays) / self.cfg.n_rays
        self._ray_phi = phi
        hx, hy = CHASSIS_HALF
        with np.errstate(divide="ignore"):
            edge = np.minimum(hx / np.abs(np.cos(phi)), hy / np.abs(np.sin(phi)))
        self._ray_start = edge + 0.02

    def _limits(self):
        # (x_min, x_max, y_min, y_max) the base joints allow - bounding box inset, as in main.py.
        lo, hi = self._poly.min(axis=0), self._poly.max(axis=0)
        i = self.cfg.base_inset
        return lo[0] + i, hi[0] - i, lo[1] + i, hi[1] - i

    def _inside(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self._poly)

    def _collides(self, margin: float) -> bool:
        for body in self._obstacle_bodies():
            if p.getClosestPoints(self.robot.id, body, margin):
                return True
        return False

    def _goal_metrics(self):
        # (distance to the goal point, wrapped bearing error, stop-aware signed turn to face it).
        dx, dy = self.goal_xy[0] - self.x, self.goal_xy[1] - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return dist, 0.0, 0.0
        bearing_abs = math.atan2(dy, dx)
        return dist, wrap_angle(bearing_abs - self.yaw), feasible_turn(self.yaw, bearing_abs)

    def _band_excess(self, dist: float) -> float:
        # How far outside the goal band the base is (0 inside it).
        lo, hi = self.band
        return max(lo - dist, dist - hi, 0.0)

    def _sense(self) -> np.ndarray:
        # Clearance (m) from the chassis outline to the nearest wall/can in each direction.
        theta = self.yaw + self._ray_phi
        d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        origin = np.array([self.x, self.y])
        start = origin + d * self._ray_start[:, None]
        end = start + d * self.cfg.ray_range
        frm = np.column_stack([start, np.full(len(d), RAY_HEIGHT)]).tolist()
        to = np.column_stack([end, np.full(len(d), RAY_HEIGHT)]).tolist()
        hits = p.rayTestBatch(frm, to)
        return np.array([h[2] for h in hits], dtype=np.float64) * self.cfg.ray_range

    def _observation(self, clearance: np.ndarray, dist: float, bearing: float, turn: float):
        x0, x1, y0, y1 = self._limits()
        limits = [min(max(v, 0.0) / 2.0, 1.0) for v in (self.x - x0, x1 - self.x, self.y - y0, y1 - self.y)]
        obs = np.concatenate([
            [min(dist / 8.0, 1.0), min(dist, 1.0),
             math.sin(bearing), math.cos(bearing), turn / (2.0 * math.pi),
             self.yaw / math.pi, float(self.phase) * 2.0 - 1.0,
             self._prev_action[0], self._prev_action[1],
             self.band[0] / 2.0, self.band[1] / 2.0],
            limits,
            clearance / self.cfg.ray_range,
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _success(self, dist: float) -> bool:
        return self._band_excess(dist) == 0.0

    def _propose(self, action):
        # Turn a policy action into the next base pose, clamped like the real joints.
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
        forward = a[0] * (c.max_forward if a[0] >= 0 else c.max_reverse)
        yaw = float(np.clip(self.yaw + a[1] * c.max_turn, -YAW_LIMIT, YAW_LIMIT))  # joint stops - no wrap
        x0, x1, y0, y1 = self._limits()
        x = float(np.clip(self.x + forward * math.cos(yaw), x0, x1))  # the base just stops at a limit
        y = float(np.clip(self.y + forward * math.sin(yaw), y0, y1))
        return a, x, y, yaw


class RLMotionEnv(_MotionCore, gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[MotionRLConfig] = None, *, gui: bool = False,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = config or MotionRLConfig()
        if self.cfg.phase not in ("outbound", "return", "mixed"):
            raise ValueError("phase must be 'outbound', 'return' or 'mixed'")
        self.rng = np.random.default_rng(seed)
        self.sampler = RoomSampler(l_shape_probability=self.cfg.l_shape_probability)

        self.env = PaintEnvironment(gui=gui, paint=PaintConfig(render_paint=False),
                                    room=self._sample_room())
        # Generous base limits - walls, not joint limits, are what stop the chassis.
        self.robot = Robot(start_xy=(1.0, 1.0), x_range=(-1.0, 8.0), y_range=(-1.0, 8.0))
        self.can = PaintCan(position=(0.0, 0.0, 0.0))
        self._stow()
        self._poly = np.array(self.env.room.footprint(), float)
        self._init_rays()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs_dim = 15 + self.cfg.n_rays
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._episodes = 0
        self.x = self.y = self.yaw = 0.0
        self.phase = OUTBOUND
        self.goal_xy = (0.0, 0.0)
        self.band = self.cfg.dip_band
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._path = 0.0
        self._inset_steps = 0
        self._prev_excess = 0.0

    # ------------------------------------------------------------------ robot / scene helpers
    def _stow(self):
        # Arm vertical + fully retracted (see STOW_TILT) - the roller must end up overhead.
        for idx, v in zip(self.robot.joint_indices[3:], stowed_joints(0, 0, 0)[3:]):
            p.resetJointState(self.robot.id, idx, v)
        if self.robot.get_roller_position()[2] < 1.0:
            raise RuntimeError("STOW_TILT does not put the roller overhead; check the sign")

    def _apply_pose(self, x: float, y: float, yaw: float):
        self.x, self.y, self.yaw = x, y, yaw
        for idx, v in zip(self.robot.joint_indices[:3], (x, y, yaw)):
            p.resetJointState(self.robot.id, idx, v)

    def _sample_room(self):
        room = self.sampler.sample(seed=int(self.rng.integers(2 ** 31 - 1)))
        return room if self.cfg.keep_openings else replace(room, openings={})

    def _rebuild_room(self):
        self.env.reset(room=self._sample_room())
        self._poly = np.array(self.env.room.footprint(), float)

    def _obstacle_bodies(self):
        return [w.body_id for w in self.env.walls] + list(self.can.collision_body_ids)

    def _pose_is_free(self, x: float, y: float, yaw: float, margin: float) -> bool:
        x0, x1, y0, y1 = self._limits()
        if not (x0 <= x <= x1 and y0 <= y <= y1) or not self._inside(x, y):
            return False
        self._apply_pose(x, y, yaw)
        return not self._collides(margin)

    def _move_can(self, x: float, y: float):
        self.can.position = [x, y, 0.0]
        p.resetBasePositionAndOrientation(self.can.body_id, [x, y, 0.0], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(
            self.can.bottom_id, [x, y, self.can.wall_thickness / 2], [0, 0, 0, 1])

    # ------------------------------------------------------------------ episode generation
    def _bbox(self):
        return self._poly.min(axis=0), self._poly.max(axis=0)

    def _place_can(self) -> bool:
        lo, hi = self._bbox()
        for _ in range(200):
            x, y = self.rng.uniform(lo[0], hi[0]), self.rng.uniform(lo[1], hi[1])
            if (self._inside(x, y) and
                    distance_to_polygon_edges(x, y, self._poly) >= self.cfg.can_wall_clearance):
                self._move_can(x, y)
                return True
        return False

    def _random_wall_point(self):
        # (point on a random wall, inward normal), chosen along the wall away from its ends.
        specs = self.env.room.wall_specs()
        lengths = np.array([s[4] for s in specs], float)
        _, origin, u_dir, normal, length, _ = specs[int(self.rng.choice(len(specs), p=lengths / lengths.sum()))]
        margin = 0.6
        u = self.rng.uniform(margin, length - margin) if length > 2 * margin else length / 2
        return np.array([origin[0] + u * u_dir[0], origin[1] + u * u_dir[1]]), normal[:2]

    def _painting_like_pose(self):
        # In front of a wall, 0.8-1.6 m out, any heading - roughly where the base stands while painting.
        lo, hi = self.cfg.start_wall_range
        for _ in range(60):
            point, normal = self._random_wall_point()
            s = self.rng.uniform(lo, hi)
            x, y = point[0] + normal[0] * s, point[1] + normal[1] * s
            yaw = float(self.rng.uniform(-YAW_LIMIT, YAW_LIMIT))
            if self._pose_is_free(x, y, yaw, self.cfg.spawn_margin):
                return x, y, yaw
        return None

    def _random_free_pose(self):
        lo, hi = self._bbox()
        for _ in range(200):
            x, y = self.rng.uniform(lo[0], hi[0]), self.rng.uniform(lo[1], hi[1])
            yaw = float(self.rng.uniform(-YAW_LIMIT, YAW_LIMIT))
            if self._pose_is_free(x, y, yaw, self.cfg.spawn_margin):
                return x, y, yaw
        return None

    def _ring_pose(self):
        # Around the can, where a dip leaves the base, with any heading.
        cx, cy = self.can.position[0], self.can.position[1]
        lo, hi = self.cfg.return_start_range
        for _ in range(60):
            a = float(self.rng.uniform(-math.pi, math.pi))
            r = float(self.rng.uniform(lo, hi))
            yaw = float(self.rng.uniform(-YAW_LIMIT, YAW_LIMIT))
            if self._pose_is_free(cx + r * math.cos(a), cy + r * math.sin(a), yaw, self.cfg.spawn_margin):
                return cx + r * math.cos(a), cy + r * math.sin(a), yaw
        return None

    def _sample_episode(self) -> bool:
        if not self._place_can():
            return False
        if self.cfg.phase == "mixed":
            self.phase = OUTBOUND if self.rng.random() < 0.5 else RETURN
        else:
            self.phase = OUTBOUND if self.cfg.phase == "outbound" else RETURN
        can_xy = self.can.position[:2]

        if self.phase == OUTBOUND:
            start = None
            if self.rng.random() >= self.cfg.start_anywhere_prob:
                start = self._painting_like_pose()
            start = start or self._random_free_pose()
            self.goal_xy, self.band = tuple(can_xy), self.cfg.dip_band
        else:
            start = self._ring_pose()
            point, _ = self._random_wall_point()
            z = float(self.rng.uniform(*self.cfg.target_z_range))
            self.goal_xy = (float(point[0]), float(point[1]))
            self.band = paint_band(z, self.cfg.band_margin)
        if start is None:
            return False
        self._apply_pose(*start)
        # An episode that starts inside the goal band teaches nothing.
        return not self._success(math.dist(start[:2], self.goal_xy))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._episodes > 0 and self._episodes % self.cfg.rebuild_room_every == 0:
            self._rebuild_room()
        self._episodes += 1
        for _ in range(100):
            if self._sample_episode():
                break
            self._rebuild_room()  # an unusable room (e.g. too tight for the can) - draw another
        else:
            raise RuntimeError("could not sample a valid motion-RL episode")

        self._steps = 0
        self._path = 0.0
        self._inset_steps = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        dist, bearing, turn = self._goal_metrics()
        self._prev_excess = self._band_excess(dist)
        return self._observation(self._sense(), dist, bearing, turn), {"phase": self.phase}

    def step(self, action):
        c = self.cfg
        a, x, y, yaw = self._propose(action)
        moved = math.hypot(x - self.x, y - self.y)
        self._apply_pose(x, y, yaw)
        self._steps += 1
        self._path += moved

        out_of_bounds = not self._inside(x, y)
        collision = (not out_of_bounds) and self._collides(c.collision_margin)

        clearance = self._sense()
        dist, bearing, turn = self._goal_metrics()
        excess = self._band_excess(dist)
        success = (not collision and not out_of_bounds) and excess == 0.0

        reward = c.w_progress * (self._prev_excess - excess)
        reward -= c.w_time + c.w_action * float(a @ a)
        reward -= c.w_clearance * max(0.0, (c.clearance_soft - clearance.min()) / c.clearance_soft)
        wall_dist = distance_to_polygon_edges(x, y, self._poly)
        too_close = wall_dist < c.base_inset
        self._inset_steps += int(too_close)
        if too_close:
            reward -= c.w_inset * min(1.0, (c.base_inset - wall_dist) / (c.base_inset - c.inset_full_penalty))
        if success:
            reward += c.r_success
        if collision or out_of_bounds:
            reward -= c.r_collision

        self._prev_excess = excess
        self._prev_action = a.astype(np.float32)
        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(not terminated and self._steps >= c.max_steps)
        info = {"is_success": bool(success), "collision": bool(collision),
                "out_of_bounds": bool(out_of_bounds), "phase": self.phase,
                "path_length": self._path, "steps": self._steps,
                "inset_frac": self._inset_steps / self._steps}
        return self._observation(clearance, dist, bearing, turn), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


class BaseNavigator(_MotionCore):
    # Runs a trained policy against the LIVE robot - it only decides the next base pose. The
    # controller does the stowing and executes each pose through the normal joint controller, so
    # collision monitoring during execution is unchanged.
    #
    #   nav.start(OUTBOUND, can_xy, cfg.dip_band)        # drive into the ring around the can
    #   nav.start(RETURN, wall_xy, nav.paint_band(z))    # drive into the band in front of a target
    #   state, pose = nav.step()                         # "continue" + (x, y, yaw) to command,
    #                                                    # or "success" / "collision" / "timeout"

    def __init__(self, model, robot, wall_bodies, can, footprint, cfg: Optional[MotionRLConfig] = None):
        self.cfg = cfg or MotionRLConfig()
        self.model = model
        self.robot = robot
        self.can = can
        self._bodies = list(wall_bodies) + list(can.collision_body_ids)
        self._poly = np.array(footprint, float)
        self._init_rays()
        self.x = self.y = self.yaw = 0.0
        self.phase = OUTBOUND
        self.goal_xy = (0.0, 0.0)
        self.band = self.cfg.dip_band
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._steps = 0

    @classmethod
    def load(cls, model_path, robot, wall_bodies, can, footprint, cfg: Optional[MotionRLConfig] = None):
        # macOS - torch's OpenMP threads can take down PyBullet's GUI thread on the first prediction
        # (same workaround as the coverage planner). The settings must be in place before torch is
        # imported, and they are made here, in the only place that needs torch, so that runs which
        # never load a policy do not import torch (or touch these variables) at all.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        model = PPO.load(model_path, device="cpu")
        expected = 15 + (cfg or MotionRLConfig()).n_rays
        if model.observation_space.shape != (expected,):
            raise ValueError(
                f"{model_path} was trained with a {model.observation_space.shape[0]}-value observation, "
                f"but this code builds {expected}. Models trained before the band-goal change "
                f"(everything under models/motion_rl*) must be retrained.")
        return cls(model, robot, wall_bodies, can, footprint, cfg)

    def _obstacle_bodies(self):
        return self._bodies

    def read_pose(self):
        joints = self.robot.get_joint_states()
        self.x, self.y, self.yaw = float(joints[0]), float(joints[1]), float(joints[2])

    def paint_band(self, target_z: float):
        return paint_band(target_z, self.cfg.band_margin)

    def start(self, phase: int, goal_xy, band):
        self.phase = phase
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self.band = (float(band[0]), float(band[1]))
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def arrived(self) -> bool:
        self.read_pose()
        return self._success(self._goal_metrics()[0])

    def step(self):
        self.read_pose()
        self._steps += 1
        if self._collides(self.cfg.collision_margin):
            return "collision", None
        dist, bearing, turn = self._goal_metrics()
        if self._success(dist):
            return "success", None
        if self._steps > self.cfg.max_steps:
            return "timeout", None
        obs = self._observation(self._sense(), dist, bearing, turn)
        action, _ = self.model.predict(obs, deterministic=True)
        a, x, y, yaw = self._propose(action)
        self._prev_action = a.astype(np.float32)
        return "continue", (x, y, yaw)
