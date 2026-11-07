#!/usr/bin/env python3
"""Free-motion TD3 wall coverage for the uploaded telescopic painting robot.

Put this file beside main.py, with the original simulation/ and planning/ folders.
Python 3.9+; dependencies: pybullet numpy pillow gymnasium stable-baselines3 torch.

  python train_td3_wall.py --check
  python train_td3_wall.py --check-rewards
  python train_td3_wall.py --steps 300000 --out runs/td3_wall_v3
  python train_td3_wall.py --evaluate runs/td3_wall_v3/final.zip --gui --diagnostics
  python train_td3_wall.py --resume runs/td3_wall_v3/final.zip --layout random \
      --steps 300000 --out runs/td3_wall_openings

One wall per episode, selected from south/east/north/west. Training varies room
dimensions by default. Actions are velocities along the wall, up/down, toward/
away from it, and rotation of the roller axis within its plane. An analytic IK
controller moves the base/arm; it does NOT supply a painting route. Global and
local coverage maps tell the policy where unfinished surface remains.

Unlimited paint for this stage; no dipping, room navigation between walls, or
guarantee of generalization to arbitrary geometry. The root of the robot's two
translation joints is expressed in the active wall frame at reset. This changes
coordinates, not the mechanism. Motors, contact, and collisions run in Bullet.
The uploaded URDF has no collision shapes for most arm links.

Success means 98% of a conservative kinematic target mask. Its fraction of the
entire paintable wall and ACTUAL whole-wall coverage are always reported. The
mask excludes conservative floor/corner/reach margins; it is not proof that all
excluded pixels are physically unreachable. Openings do not count as wall area.

Reward v3 per control step (20 Hz): new target m2 / .01; up to 20% extra for
new paint with the roller axis perpendicular to motion. Bounded proximity
potential rewards approaching and charges retreat, without rewarding loops.
Costs: gap up to .04, missing painting contact up to .01, no new paint .008,
excess overlap up to .005, rotation command up to .005, time .002, action
change up to .008, rejected command .01. +30 success; -20 stall; -120 collision.
All painting/contact measurements use actual physics; proximity cannot paint.
Stationary contact and repeated repainting cannot earn recurring positive reward.
These weights are a starting configuration, not proof of successful learning.
Default stall deadlines: 15 s idle contact, 45 s without meaningful new area.
Episode length: up to 30,000 control steps, not a small patch's 1-minute limit.

Controller v2: motor lag is recovered with bounded, coordinated joint targets;
it no longer causes every subsequent action to be rejected. Cartesian commands
pause while the motor catches up. Unreachable proposals are shortened, with the
last feasible pose still actively controlled. Contact requires actual Bullet
contact; it has NOT been replaced by a distance-only painting rule. Reset lets
the robot settle and checks actual contact before starting an episode.

Lifecycle fix (v2.1): reuse one physics connection across episode resets; clear
the world with resetSimulation and rebuild all robot/body references. The GUI
window stays open until close(). --check runs the Gym API checks headlessly,
then optional visual checks in one GUI session. Unexpected Bullet errors keep
their full traceback; cleanup is no longer misreported as a user-closed window.
Observations/actions and controller remain compatible with v2 models; rewards change.

This file uses a 4-action Dict-observation policy. Old patch/sweep checkpoints
are incompatible. Whole-wall v1/v2 checkpoints can be evaluated with v3 rewards,
or used with --warm-start (actor only; fresh critics/optimizers/replay). Evaluation
does not update a trained actor. Start fresh for the reported saturated policy.
Use --resume only for v3 checkpoints. Never reuse v1/v2 critics or replay.
Resume restores weights/optimizers but collects a fresh replay buffer unless
--replay-buffer is supplied. Preserve config.json beside checkpoints for loading.
--check uses short scripted strokes for diagnostics only, never training replay.
With --gui, live reward, episode return, every reward term, coverage and contact
appear at the top left of the simulation. The display follows camera changes
on each control step and resets with each episode. Existing v3 models work.

API references:
https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
Potential shaping: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import sys
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import pybullet as p
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.paint_environment import PaintEnvironment, PaintConfig
from simulation.paint_wall import Wall, Opening
from simulation.robot import Robot
from planning.motion_planner import MotionPlanner


VERSION = "td3-whole-wall-v3"
LEGACY_VERSIONS = ("td3-whole-wall-v1", "td3-whole-wall-v2")
GAMMA = .999
WALLS = ("south", "east", "north", "west")
DT = 1 / 20
SUBSTEPS = 12
PPM = 100
RADIUS, HALF_LENGTH, INSET = .06, .09, .50
RESAMPLE = getattr(Image, "Resampling", Image)


@dataclass
class Config:
    walls: tuple = WALLS
    layout: str = "solid"
    randomize_size: bool = True
    max_episode_steps: int = 30000
    stall_seconds: float = 45.0
    idle_contact_seconds: float = 15.0
    success_fraction: float = .98

    def validate(self):
        if not self.walls or any(w not in WALLS for w in self.walls):
            raise ValueError("Select at least one of south east north west")
        if self.layout not in ("solid", "original", "random"):
            raise ValueError("Unknown layout")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.stall_seconds, self.idle_contact_seconds)):
            raise ValueError("Stall times must be finite and nonnegative; zero disables")


def center_bounds(length, height, theta):
    # Bounding box of the cylinder, including its radius (not just the axis line).
    eu = HALF_LENGTH * abs(math.cos(theta)) + RADIUS * abs(math.sin(theta))
    ev = HALF_LENGTH * abs(math.sin(theta)) + RADIUS * abs(math.cos(theta))
    return (eu + .008, length - eu - .008, ev + .012, height - ev - .004)


def reach_parameters(u, v, normal, length, depth):
    """Vectorized analytic reach test, in the active wall's local frame."""
    u, v = np.asarray(u), np.asarray(v)
    base_u = np.clip(u, INSET, length - INSET)
    sideways = u - base_u
    dz = v - 1.0
    min_reach, max_reach = .86 + .095 + .001, .86 + .85 - .001
    hmin = np.maximum(np.sqrt(np.maximum(0, min_reach**2 - dz**2)),
                      np.hypot(sideways, INSET - normal) - .02)
    hmax = np.minimum(np.sqrt(np.maximum(0, max_reach**2 - dz**2)),
                      np.hypot(sideways, depth - INSET - normal) - .02)
    valid = (np.abs(dz) < max_reach) & (hmin <= hmax) & (hmax > 0)
    preferred = np.sqrt(np.maximum(0, 1.15**2 - dz**2))
    horizontal = np.minimum(np.maximum(preferred, hmin), hmax)
    base_n = np.clip(normal + np.sqrt(np.maximum(0, (horizontal + .02)**2 - sideways**2)),
                     INSET, depth-INSET)
    return valid, base_u, base_n, horizontal


def solve_joints(command, length, depth, previous=None):
    """Exact inverse for this robot's 0.86 m fixed reach, shoulder z=1 m.

    The base translation axes are mounted in the wall frame once at reset.
    Positive normal points INTO the room, so yaw stays near -pi/2 for all walls.
    """
    u, v, normal, theta = map(float, command)
    ok, bu, bn, horizontal = reach_parameters(u, v, normal, length, depth)
    if not bool(ok):
        return None
    yaw = math.atan2(normal - float(bn), u - float(bu))
    pitch = math.atan2(1 - v, float(horizontal))
    extension = math.hypot(float(horizontal), v - 1) - .86
    # Transform the requested wall-plane axis into the frame before the wrist.
    axis = np.array([math.cos(theta), 0., math.sin(theta)])
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    local = np.array([cy * axis[0], -sy * axis[0], axis[2]])
    local = np.array([cp * local[0] - sp * local[2], local[1],
                      sp * local[0] + cp * local[2]])
    candidates = []
    # Both signs of a cylinder axis describe the same roller. Choose nearby joints.
    for sign in (1, -1):
        dx, dy, dz = sign * local
        principal = math.asin(float(np.clip(dx, -1, 1)))
        for wrist in (principal, math.pi - principal, -math.pi - principal):
            if abs(wrist) > 3.14159:
                continue
            cosine = math.cos(wrist)
            roll = (float(previous[5]) if previous is not None else 0.) if abs(cosine) < 1e-8 else math.atan2(-dy / cosine, dz / cosine)
            if abs(roll) > 3.14159:
                continue
            q = np.array([bu, bn, yaw, pitch, extension, roll, wrist], float)
            score = np.sum((q[5:] - previous[5:])**2) if previous is not None else roll**2 + wrist**2
            candidates.append((score, q))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def target_mask(canvas, depth):
    """Conservative union of feasible roller-axis footprints, eight orientations.

    For every wall pixel, find an axis centre that covers it and respects the
    cylinder's edge clearances. Check arm/base reach at that centre. This is a
    kinematic estimate; --check additionally probes real Bullet poses.
    """
    rows, cols = np.indices(canvas.paintable.shape)
    u = canvas.length - (cols + .5) / canvas.ppm
    v = canvas.height - (rows + .5) / canvas.ppm
    result = np.zeros_like(canvas.paintable)
    for theta in np.arange(8) * math.pi / 8:
        ax, az = math.cos(theta), math.sin(theta)
        u0, u1, v0, v1 = center_bounds(canvas.length, canvas.height, theta)
        low = np.full_like(u, -HALF_LENGTH)
        high = np.full_like(u, HALF_LENGTH)
        for coordinate, component, lower, upper in ((u, ax, u0, u1), (v, az, v0, v1)):
            if abs(component) < 1e-8:
                high = np.where((coordinate >= lower) & (coordinate <= upper), high, -1.)
            else:
                t0, t1 = (coordinate - lower) / component, (coordinate - upper) / component
                low = np.maximum(low, np.minimum(t0, t1))
                high = np.minimum(high, np.maximum(t0, t1))
        t = np.clip(np.zeros_like(u), low, np.maximum(low, high))
        reachable, _, _, _ = reach_parameters(u - ax * t, v - az * t, .0595,
                                               canvas.length, depth)
        result |= (low <= high) & reachable
    return result & canvas.paintable



def stage3_sampled_ik_mask(env, spacing=.025):
    """Stage-3-only candidate paint mask from IK-feasible continuous sweep segments.

    This is NOT a proof of physical trajectory/contact reachability. It uses the
    environment's exact IK and joint limits, and marks roller-axis footprints
    swept between neighboring feasible centers. A separate Bullet trajectory
    validation remains mandatory. Never apply to Stages 1 or 2.
    """
    from PIL import ImageDraw

    c = env.wall.canvas
    result = Image.new('1', (c.width_px, c.height_px), 0)
    draw = ImageDraw.Draw(result)
    stats = {}
    for theta, travel_axis in ((0., 1), (math.pi / 2, 0)):
        u0, u1, v0, v1 = center_bounds(c.length, c.height, theta)
        us = np.linspace(u0, u1, max(2, math.ceil((u1-u0)/spacing)+1))
        vs = np.linspace(v0, v1, max(2, math.ceil((v1-v0)/spacing)+1))
        feasible = np.zeros((len(vs), len(us)), dtype=bool)
        for j, v in enumerate(vs):
            for i, u in enumerate(us):
                feasible[j, i] = env._ik(np.array([u, v, .0595, theta])) is not None
        stats[str(int(round(math.degrees(theta))))] = {
            'sampled_centers': int(feasible.size),
            'ik_feasible_centers': int(feasible.sum()),
        }
        def axis_ends(u, v):
            a = np.array([u - HALF_LENGTH*math.cos(theta),
                          v - HALF_LENGTH*math.sin(theta)])
            b = np.array([u + HALF_LENGTH*math.cos(theta),
                          v + HALF_LENGTH*math.sin(theta)])
            return tuple(map(float, c.uv_to_px(*a))), tuple(map(float, c.uv_to_px(*b)))
        # The real paint model uses a narrow stroke, not a 12-cm roller diameter.
        stroke_px = max(1, round(env.sim.paint.patch_width*c.ppm))
        for j, v in enumerate(vs):
            for i, u in enumerate(us):
                if not feasible[j, i]:
                    continue
                a, b = axis_ends(u, v)
                draw.line([a, b], fill=1, width=stroke_px)
                # Interpolate only the adjacent center in the travel direction.
                # Both endpoints must satisfy the robot's exact IK/joint limits.
                neighbor = (j-1, i) if travel_axis == 1 else (j, i-1)
                nj, ni = neighbor
                if nj < 0 or ni < 0 or not feasible[nj, ni]:
                    continue
                aa, bb = axis_ends(us[ni], vs[nj])
                draw.polygon([aa, bb, b, a], fill=1)
    candidate = (np.asarray(result, dtype=bool) & c.paintable)
    # Never add pixels that the original conservative target excludes.
    candidate &= env.target
    if not candidate.any():
        raise RuntimeError('Stage 3 sampled IK mask is empty')
    return candidate, stats


def contact_potential(gap):
    """Bounded proximity signal using measured roller-surface clearance, metres."""
    return math.exp(-max(0., float(gap)) / .02)


def coverage_reward(new_area, overlap, footprint, alignment, action_change,
                    collision, stalled, success, rejected, *,
                    previous_gap, gap, contact_fraction, rotation_action):
    """Painting-only reward, per 50 ms action; every term is logged separately.

    Only actual, previously unpainted target pixels earn coverage reward. There
    is no recurring positive contact bonus. Proximity shaping is gamma*Phi(s')
    - Phi(s), with zero next potential on task termination, but NOT truncation.
    Its discounted sum telescopes: moving away and returning cannot farm it.
    New coverage and alignment use measured motion/contact, never commanded paint.

    Gap/contact costs intentionally apply to the painting phase only; revise
    them before adding dipping or navigation. A collision costs more than the
    bounded discounted nonterminal costs plus a stall (gamma=.999), so crashing
    is not a cheap way to escape these costs. These are starting weights, not
    a guarantee of learning or full-wall coverage.
    """
    productive = max(0., new_area) / .01
    terminal = collision or stalled or success
    next_potential = 0. if terminal else contact_potential(gap)
    return {
        "new_paint": productive,
        "orientation": .20 * productive * float(np.clip(alignment, 0, 1)),
        "approach": GAMMA * next_potential - contact_potential(previous_gap),
        "wall_gap": -.04 * float(np.clip(max(0., gap) / .05, 0, 1))**2,
        "lost_contact": -.01 * (1. - float(np.clip(contact_fraction, 0, 1))),
        "no_new_paint": -.008 if new_area < 1e-12 else 0.,
        "repaint": -min(.005, .05 * max(0., overlap - .20 * footprint) / .01),
        "time": -.002,
        "action_change": -.002 * float(np.clip(action_change, 0, 4)),
        "rotation_effort": -.005 * float(np.clip(rotation_action, -1, 1))**2,
        "rejected": -.01 if rejected else 0.,
        "failure": -120. if collision else -20. if stalled else 0.,
        "success": 30. if success and not collision and not stalled else 0.,
    }


def check_rewards():
    """Quick behavioral reward checks; not a substitute for learned evaluation."""
    base = dict(new_area=0., overlap=.0008, footprint=.0008, alignment=1.,
                action_change=0., collision=False, stalled=False,
                success=False, rejected=False, previous_gap=0., gap=0.,
                contact_fraction=1., rotation_action=0.)
    def score(**changes):
        return coverage_reward(**dict(base, **changes))
    painting = score(new_area=.0005, overlap=.0003)
    idle = score()
    air = score(overlap=0., footprint=0., previous_gap=.1,
                gap=.1, contact_fraction=0., rotation_action=1.)
    assert sum(painting.values()) > 0 > sum(idle.values()) > sum(air.values())
    assert score(new_area=.0005, alignment=1.)['orientation'] > score(
        new_area=.0005, alignment=0.)['orientation']
    assert idle['orientation'] == air['new_paint'] == 0.
    assert score(previous_gap=.03, gap=.02)['approach'] > 0
    assert score(previous_gap=.02, gap=.03)['approach'] < 0
    # Discounted retreat/return loop cannot earn positive shaping reward.
    out = score(previous_gap=0., gap=.03)['approach']
    back = score(previous_gap=.03, gap=0.)['approach']
    assert out + GAMMA * back <= 0
    assert score(stalled=True)['approach'] == -contact_potential(0.)
    assert score(collision=True)['failure'] == -120.
    # Maximum bounded ongoing costs sum to .088; add stall and potential loss.
    assert 120. > .088 / (1 - GAMMA) + 20. + 1.
    for name, terms in (("new paint", painting), ("idle contact", idle),
                        ("100 mm away + rotation", air)):
        print(f"[reward] {name}: {sum(terms.values()):+.5f} per step")
    print("[reward] PASS coverage, orientation, approach, anti-loop and failure-cost checks")


def bounded_joint_target(current, desired, dt=DT):
    """Keep a corrective motor target even when actual joints lag the IK goal.

    A common interpolation factor preserves the direction of the joint-space
    correction. This limits each motor target increment; actual contacts and
    collisions are still checked every physics step. Never resets joint state.
    """
    current, desired = np.asarray(current, float), np.asarray(desired, float)
    delta = desired-current
    limits = np.array([.60, .60, 1.20, 1.00, .60, 1.50, 1.50])*dt
    scale = float(min(1., np.min(limits/np.maximum(np.abs(delta), 1e-12))))
    return current+scale*delta, scale


class WallScene(PaintEnvironment):
    """Use the user's wall/canvas/camera implementation with varied room geometry."""
    def __init__(self, rng, config, client, gui=False):
        self.gui, self.dt = gui, 1 / 240
        self.paint = PaintConfig(px_per_m=PPM, patch_width=.004, gui_texture_every=48)
        # WholeWallEnv owns this connection. Scene reset must never disconnect it.
        self.client = client
        self._work = tempfile.TemporaryDirectory(prefix="td3_wall_")
        self._steps, self._cam_keys_active = 0, False
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        if config.randomize_size:
            self.width, self.depth, self.height = map(float,
                rng.uniform([3.6, 2.8, 2.2], [4.4, 3.6, 2.6]))
        else:
            self.width, self.depth, self.height = 4., 3., 2.6
        W, D, H = self.width, self.depth, self.height
        half = [W / 2 + .5, D / 2 + .5, .05]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[.55, .42, .3, 1])
        self.floor_id = p.createMultiBody(0, col, vis, basePosition=[W / 2, D / 2, -.05])
        specs = [("south", (0, 0, 0), (1, 0, 0), (0, 1, 0), W, (.1, .1)),
                 ("east", (W, 0, 0), (0, 1, 0), (-1, 0, 0), D, (0, 0)),
                 ("north", (W, D, 0), (-1, 0, 0), (0, -1, 0), W, (.1, .1)),
                 ("west", (0, D, 0), (0, -1, 0), (1, 0, 0), D, (0, 0))]
        self.walls = []
        for name, origin, u, normal, length, ext in specs:
            holes = ()
            if config.layout == "original":
                # Exact original openings at 4 x 3 x 2.6; scale with other sizes.
                base = {"south": (2.7/4, .9/2.6, .9/4, 1/2.6),
                        "east": (.4/3, 0, .9/3, 2.05/2.6),
                        "north": (1.4/4, .9/2.6, 1.2/4, 1/2.6)}.get(name)
                if base:
                    holes = (Opening(base[0]*length, base[1]*H, base[2]*length, base[3]*H),)
            elif config.layout == "random" and rng.random() < .75:
                width, height = float(rng.uniform(.55, 1.0)), float(rng.uniform(.65, 1.15))
                u0 = float(rng.uniform(.3, length - width - .3))
                v0 = 0. if rng.random() < .3 else float(rng.uniform(.45, H - height - .15))
                if v0 == 0:
                    height = min(2.05, H - .2)
                holes = (Opening(u0, v0, width, height),)
            wall = Wall(name, np.array(origin), np.array(u), np.array(normal), length, H,
                        .1, holes, ext)
            wall.spawn(self._work.name, PPM, (235, 232, 222), (40, 90, 200), (.92, .91, .87, 1))
            self.walls.append(wall)
        self.walls_by_name = {w.name: w for w in self.walls}
        self.CAMERA_VIEWS = {
            1: (max(W, H)*.9, 180, -8, (W/2, 0, H/2)),
            2: (max(D, H)*1.1, 270, -8, (W, D/2, H/2)),
            3: (max(W, H)*.9, 0, -8, (W/2, D, H/2)),
            4: (max(D, H)*1.1, 90, -8, (0, D/2, H/2)),
            5: (max(W, D)*1.4, 165, -40, (W/2, D/2, .5))}
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self._cam_view = self.CAMERA_VIEWS[5]
            self._apply_cam()

    def close(self):
        """Release only this scene's files; the environment owns the connection."""
        self._work.cleanup()


class WholeWallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    GLOBAL, LOCAL, STATE = 24, 16, 40

    def __init__(self, config=None, gui=False):
        super().__init__()
        self.config = config or Config()
        self.config.validate()
        self.gui, self.render_mode = gui, "human" if gui else None
        self.sim, self.done = None, True
        self._client = None
        self.robot = self.planner = self.wall = None
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (self.STATE,), np.float32),
            "global_map": spaces.Box(0, 1, (3*self.GLOBAL**2,), np.float32),
            "local_map": spaces.Box(0, 1, (3*self.LOCAL**2,), np.float32)})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        name = (options or {}).get("wall", str(self.np_random.choice(self.config.walls)))
        if name not in self.config.walls:
            raise ValueError("Requested wall is outside config.walls")
        self._reset_world()
        self.sim = getattr(self, "scene_factory", WallScene)(self.np_random, self.config, client=self._client, gui=self.gui)
        self.wall = self.sim.walls_by_name[name]
        self.depth = self.sim.depth if name in ("south", "north") else self.sim.width
        self.robot = getattr(self, "robot_factory", Robot)(start_xy=(self.wall.length/2, 1.),
                           x_range=(INSET, self.wall.length-INSET),
                           y_range=(INSET, self.depth-INSET))
        self._validate_robot()
        # Translation joint coordinates become (wall u, inward n), for every wall.
        p.resetBasePositionAndOrientation(self.robot.id, self.wall.origin.tolist(),
                                          p.getQuaternionFromEuler([0, 0, self.wall.yaw]))
        self.planner = MotionPlanner(self.robot, [w.body_id for w in self.sim.walls] + [self.sim.floor_id]
                                     + list(getattr(self.sim, "extra_collision_bodies", ())))
        self.planner.configure_environment(self.sim)
        self.target = target_mask(self.wall.canvas, self.depth)
        self.target_pixels = int(self.target.sum())
        if self.target_pixels == 0:
            raise RuntimeError("Empty target mask for this robot/room")
        self.target_area = self.target_pixels / PPM**2
        self.previous_action = np.zeros(4, np.float32)
        self.steps, self.last_progress_step, self.idle_steps = 0, 0, 0
        self.area, self.meaningful_area = 0., 0.
        self.contact, self.done, self.reason = False, False, "running"
        self.reward_metrics = {}
        self.episode_return = 0.
        self.hud_ids = []
        self.command_status, self.servo_scale = "reset", 1.
        self.rejected_steps, self.recovery_steps = 0, 0
        self.joint_target_error = np.zeros(7)
        self.goal_uv = np.zeros(2)
        # Random start on actual solid wall surface, with either useful orientation.
        specified = (options or {}).get("command")
        for _ in range(100):
            self.command = np.array(specified if specified is not None else
                [self.np_random.uniform(.25, self.wall.length-.25),
                 self.np_random.uniform(.30, min(1.8, self.wall.height-.2)),
                 .0595, self.np_random.choice([0., math.pi/2])], float)
            q = self._ik(self.command)
            if q is None or not self._solid_at(self.command[:2]):
                if specified is not None:
                    raise ValueError("Diagnostic start pose is not feasible and paintable")
                continue
            self.robot.set_joint_targets(q, instant=True)  # Reset only.
            p.performCollisionDetection()
            self.verify_pose(self.command)
            if self.planner.current_collision(self.wall.body_id, executing=True):
                if specified is not None:
                    raise RuntimeError("Diagnostic start pose is in collision")
                continue
            # Validate a physically settled start, not only the teleported FK pose.
            settled_contact, collision = 0, False
            for _ in range(120):
                self.robot.set_joint_targets(q)
                self.sim.step()
                collision = self.planner.current_collision(self.wall.body_id, executing=True)
                if collision:
                    break
                settled_contact = settled_contact+1 if self._contact_state()["paint_contact"] else 0
                if settled_contact >= 12:  # 50 ms of consecutive valid contacts.
                    break
            if not collision and settled_contact >= 12:
                self.contact = True
                break
            if specified is not None:
                raise RuntimeError("Diagnostic start did not establish stable painting contact: "
                                   + json.dumps(self._contact_state()))
        else:
            raise RuntimeError("Could not establish stable painting contact; run --check --gui. "
                               + json.dumps(self._contact_state()))
        self.wall.canvas.lift()  # Settling never paints or earns reward.
        self._update_goal()
        if self.gui:
            self.sim.focus_camera_on_wall(name)
            self._draw_reward_hud(self._info(), 0.)
        return self._obs(), self._info()

    def _reset_world(self):
        """Reuse the owned connection; discard every old body/link reference first."""
        self.done = True
        if self._client is None:
            if p.isConnected():
                raise RuntimeError("Run only one PyBullet environment per process")
            self._client = p.connect(p.GUI if self.gui else p.DIRECT)
            if self._client < 0:
                self._client = None
                raise RuntimeError("Could not open PyBullet")
            # The supplied Robot/Wall/Planner modules use Bullet's default client 0.
            if self._client != 0:
                client, self._client = self._client, None
                p.disconnect(client)
                raise RuntimeError("The supplied simulation modules require physics client 0; use a fresh process")
        elif not p.isConnected(self._client):
            raise RuntimeError("PyBullet connection was closed before reset; restart the command")
        old_scene, self.sim = self.sim, None
        self.robot = self.planner = self.wall = None
        try:
            p.resetSimulation(physicsClientId=self._client)
        finally:
            # Remove files after Bullet has released its old shapes/textures.
            if old_scene is not None:
                old_scene.close()

    def _validate_robot(self):
        """Validate freshly created IDs before any roller-pose query."""
        client, body, link = self._client, self.robot.id, self.robot.roller_link_index
        if not p.isConnected(client):
            raise RuntimeError(f"Physics connection {client} disappeared while constructing the robot")
        try:
            count = p.getNumJoints(body, physicsClientId=client)
            if not 0 <= link < count:
                raise RuntimeError(f"Invalid roller link: client={client}, body={body}, link={link}, joints={count}")
            name = p.getJointInfo(body, link, physicsClientId=client)[12].decode("utf-8")
            if name != "roller_link":
                raise RuntimeError(f"Expected roller_link but found {name!r}; check simulation/robot.py")
        except p.error as exc:
            raise RuntimeError(f"Robot validation failed: client={client}, body={body}, roller_link={link}") from exc

    def _ik(self, command, previous=None):
        u0, u1, v0, v1 = center_bounds(self.wall.length, self.wall.height, command[3])
        if not (u0 <= command[0] <= u1 and v0 <= command[1] <= v1 and .0595 <= command[2] <= .16):
            return None
        q = solve_joints(command, self.wall.length, self.depth, previous)
        if q is None or np.any(q < self.robot.lower_limits) or np.any(q > self.robot.upper_limits):
            return None
        return q

    def _pose(self):
        if self.robot is None:
            raise RuntimeError("Robot pose requested before reset completed or after close()")
        try:
            pos = np.asarray(self.robot.get_roller_position())
            a, b = map(np.asarray, self.robot.get_roller_axis_segment())
        except p.error as exc:
            raise RuntimeError(
                f"Roller pose query failed: client={self._client}, body={self.robot.id}, "
                f"roller_link={self.robot.roller_link_index}; see the original Bullet error below") from exc
        axis = (a - b) / self.robot.ROLLER_LENGTH
        u, v = self.wall.world_to_uv(pos)
        local_axis = np.array([axis @ self.wall.u_dir, axis[2], axis @ self.wall.normal])
        return np.array([u, v, self.wall.signed_distance(pos)]), local_axis, a, b

    def verify_pose(self, command):
        actual, axis, _, _ = self._pose()
        wanted_axis = np.array([math.cos(command[3]), math.sin(command[3]), 0.])
        if np.linalg.norm(actual-command[:3]) > .002 or abs(axis @ wanted_axis) < .999:
            raise RuntimeError("URDF differs from the supplied robot: IK pose validation failed")

    def _solid_at(self, uv):
        c = self.wall.canvas
        x, y = map(int, c.uv_to_px(*uv))
        return 0 <= x < c.width_px and 0 <= y < c.height_px and bool(c.paintable[y, x])

    def _update_goal(self):
        # Observation hint only: nearest unfinished pixel. No route/reward target.
        rows, cols = np.nonzero(self.target & ~self.wall.canvas.painted)
        if not len(rows):
            self.goal_uv = self._pose()[0][:2]
            return
        stride = max(1, len(rows)//4000)
        rows, cols = rows[::stride], cols[::stride]
        uv = np.column_stack([self.wall.length-(cols+.5)/PPM, self.wall.height-(rows+.5)/PPM])
        current = self._pose()[0][:2]
        self.goal_uv = uv[np.argmin(np.sum((uv-current)**2, axis=1))]

    def _obs(self):
        states = p.getJointStates(self.robot.id, self.robot.joint_indices)
        q = np.array([s[0] for s in states]); dq = np.array([s[1] for s in states])
        lo, hi = np.asarray(self.robot.lower_limits), np.asarray(self.robot.upper_limits)
        pose, axis, _, _ = self._pose()
        L, H = self.wall.length, self.wall.height
        state = np.concatenate([
            2*(q-lo)/(hi-lo)-1, dq,  # 14
            [pose[0]/L, pose[1]/H, pose[2]/.16], axis,  # 6
            [self.command[0]/L, self.command[1]/H, self.command[2]/.16,
             math.sin(2*self.command[3]), math.cos(2*self.command[3])],  # 5
            self.previous_action,  # 4
            [float(self.contact), L/5, H/3, self.depth/5, self.area/self.target_area,
             min(1., (self.steps-self.last_progress_step)*DT/max(self.config.stall_seconds, DT)),
             min(1., self.idle_steps*DT/max(self.config.idle_contact_seconds, DT)),
             self.meaningful_area/.0002],  # 8
            (self.goal_uv-pose[:2]) / [L, H], [1.]  # 3; paint availability fixed at 1
        ]).astype(np.float32)
        c = self.wall.canvas
        # Canonical map: rows increase upwards, columns along wall u.
        maps = [m[::-1, ::-1] for m in (c.painted, c.paintable, self.target)]
        global_map = np.concatenate([np.asarray(Image.fromarray(m.astype(np.uint8)*255).resize(
            (self.GLOBAL, self.GLOBAL), RESAMPLE.BOX), np.float32).ravel()/255 for m in maps])
        # Fixed metric local field (0.40 m), to retain roller-scale detail on any wall.
        offset = (np.arange(self.LOCAL)+.5-self.LOCAL/2)*.025
        cc = np.floor((pose[0]+offset)*PPM).astype(int)
        rr = np.floor((pose[1]+offset)*PPM).astype(int)
        valid = (rr[:, None] >= 0) & (rr[:, None] < c.height_px) & (cc[None, :] >= 0) & (cc[None, :] < c.width_px)
        local = [np.where(valid, m[np.clip(rr, 0, c.height_px-1)[:, None],
                                 np.clip(cc, 0, c.width_px-1)[None, :]], False) for m in maps]
        return {"state": state, "global_map": global_map.astype(np.float32),
                "local_map": np.asarray(local, np.float32).ravel()}

    def _contact_state(self):
        pose, axis, a, b = self._pose()
        points = p.getContactPoints(self.robot.id, self.wall.body_id,
                                    linkIndexA=self.robot.roller_link_index)
        distances = [self.wall.signed_distance(pt) for pt in (a, b)]
        flat = abs(axis[2]) < .08
        ends_near = all(.054 <= d <= .068 for d in distances)
        reason = ("no_physics_contact" if not points else "axis_tilt" if not flat else
                  "ends_off_wall" if not ends_near else "painting_contact")
        return {"paint_contact": bool(points) and flat and ends_near,
                "raw_contacts": len(points) if points is not None else 0,
                "contact_reason": reason,
                "gap_mm": float((pose[2]-RADIUS)*1000),
                "endpoint_gaps_mm": [float((d-RADIUS)*1000) for d in distances],
                "axis_tilt_deg": math.degrees(math.asin(float(np.clip(abs(axis[2]), 0, 1))))}

    def _paint(self, draw):
        c = self.wall.canvas
        # True roller contact AND both ends near wall; no painting from a remote pose.
        if not self._contact_state()["paint_contact"]:
            c.lift()
            return False
        _, _, a, b = self._pose()
        auv, buv = self.wall.world_to_uv(a), self.wall.world_to_uv(b)
        apx, bpx = c.uv_to_px(*auv), c.uv_to_px(*buv)
        if c._prev is not None:
            pa, pb = c._prev
            # Match interchangeable cylinder ends before joining swept footprints.
            if np.linalg.norm(np.subtract(apx, pa)) > np.linalg.norm(np.subtract(bpx, pa)):
                apx, bpx, auv, buv = bpx, apx, buv, auv
            if max(np.linalg.norm(np.subtract(apx, pa)), np.linalg.norm(np.subtract(bpx, pb))) > .04*PPM:
                c.lift()  # Never paint a bridge after a discontinuity.
            else:
                draw.polygon([pa, pb, bpx, apx], fill=255)
        draw.line([apx, bpx], fill=255, width=max(1, round(self.sim.paint.patch_width*PPM)))
        c.stroke(auv, buv, self.sim.paint.patch_width)
        return True

    def _drive(self, action):
        """Accept feasible motion, recover tracking lag, and always drive the motors."""
        pose, axis, _, _ = self._pose()
        qnow = np.asarray(self.robot.get_joint_states())
        desired_axis = np.array([math.cos(self.command[3]), math.sin(self.command[3]), 0.])
        lagging = (np.linalg.norm(pose-self.command[:3]) > .030 or
                   abs(float(axis @ desired_axis)) < math.cos(.15))
        qgoal = self._ik(self.command, qnow)
        if qgoal is None:
            raise RuntimeError("Last feasible controller command has no IK solution")
        rejected = False
        if lagging:
            # Anti-windup: do not let the Cartesian goal run away from the robot.
            # Unlike v1, continue sending corrective joint targets while it catches up.
            self.command_status = "tracking_recovery"
            self.recovery_steps += 1
        else:
            accepted = False
            for fraction in (1., .5, .25, .125, .0625):
                proposed = self.command+fraction*action*np.array([.10, .10, .025, .70])*DT
                proposed[3] = (proposed[3]+math.pi) % (2*math.pi)-math.pi
                u0, u1, v0, v1 = center_bounds(self.wall.length, self.wall.height, proposed[3])
                proposed[:3] = np.clip(proposed[:3], [u0, v0, .0595], [u1, v1, .16])
                candidate = self._ik(proposed, qnow)
                if candidate is not None:
                    self.command, qgoal = proposed, candidate
                    self.command_status = "accepted" if fraction == 1. else "shortened"
                    accepted = True
                    break
            if not accepted:
                rejected = True
                self.rejected_steps += 1
                self.command_status = "unreachable"
        self.joint_target_error = qgoal-qnow
        motor_target, self.servo_scale = bounded_joint_target(qnow, qgoal)
        self.robot.set_joint_targets(motor_target)  # Always drive, including recovery.
        return rejected

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before step()")
        if not p.isConnected(self.sim.client):
            raise RuntimeError("PyBullet window closed")
        action = np.asarray(action, np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Action must contain four finite numbers")
        action = np.clip(action, -1, 1)
        c = self.wall.canvas
        before = c.painted
        start, _, _, _ = self._pose()
        rejected = self._drive(action)
        touched = Image.new("L", (c.width_px, c.height_px), 0)
        draw = ImageDraw.Draw(touched)
        contact, collision = False, False
        contact_steps = 0
        for _ in range(SUBSTEPS):
            if not p.isConnected(self.sim.client):
                raise RuntimeError("PyBullet window closed")
            self.sim.step()
            if self.planner.current_collision(self.wall.body_id, executing=True):
                collision = True
                c.lift()
                break
            painted_contact = self._paint(draw)
            contact_steps += int(painted_contact)
            contact = painted_contact or contact
        self.contact = contact
        self.steps += 1
        after = c.painted
        new_area = float(((after & ~before) & self.target).sum())/PPM**2
        footprint = (np.asarray(touched)>0) & self.target
        overlap = float((footprint & before).sum())/PPM**2
        footprint_area = float(footprint.sum())/PPM**2
        self.area = float((after & self.target).sum())/PPM**2
        self.meaningful_area += new_area
        if self.meaningful_area >= .0002 - 1e-12:  # 2 cm²; tiny jitter cannot reset forever.
            self.last_progress_step, self.meaningful_area = self.steps, 0.
        self.idle_steps = self.idle_steps+1 if contact and new_area < 1e-12 else 0
        cfg = self.config
        stalled = (self.steps*DT >= 5. and (
            (cfg.stall_seconds > 0 and (self.steps-self.last_progress_step)*DT >= cfg.stall_seconds)
            or (cfg.idle_contact_seconds > 0 and self.idle_steps*DT >= cfg.idle_contact_seconds)))
        success = self.area/self.target_area >= cfg.success_fraction and not collision and not stalled
        end, axis, _, _ = self._pose()
        motion = end[:2]-start[:2]
        alignment = 0.
        if np.linalg.norm(motion) > 1e-6:
            tangent_axis = axis[:2]/max(np.linalg.norm(axis[:2]), 1e-9)
            alignment = (1-axis[2]**2)*(1-float(tangent_axis @ (motion/np.linalg.norm(motion)))**2)
        self.reward_metrics = coverage_reward(new_area, overlap, footprint_area, alignment,
            float(np.mean((action-self.previous_action)**2)), collision, stalled, success, rejected,
            previous_gap=float(start[2]-RADIUS), gap=float(end[2]-RADIUS),
            contact_fraction=contact_steps/SUBSTEPS, rotation_action=float(action[3]))
        self.previous_action = action.copy()
        # Time limit is an external training cutoff (truncation); failures terminate.
        terminated = bool(collision or stalled or success)
        truncated = bool(self.steps >= cfg.max_episode_steps and not terminated)
        self.done = terminated or truncated
        self.reason = "collision" if collision else "stalled" if stalled else "success" if success else "time_limit" if truncated else "running"
        if self.steps % 20 == 0:
            self._update_goal()
        info = self._info()
        info.update(new_area_m2=new_area, repaint_area_m2=overlap,
                    paint_contact_fraction=contact_steps/SUBSTEPS,
                    reward_terms=self.reward_metrics.copy(), rejected_command=bool(rejected))
        reward = float(sum(self.reward_metrics.values()))
        self.episode_return += reward
        info["episode_return"] = self.episode_return
        if self.gui:
            self._draw_reward_hud(info, reward)
        return self._obs(), reward, terminated, truncated, info

    def _draw_reward_hud(self, info, reward):
        """Camera-facing live text near the top left; replace items, never stack.

        Unproject screen positions onto a plane near the camera so all wall
        views stay readable. Bullet expects column-major view/projection arrays.
        This display has no effect on observations, rewards or motor commands.
        """
        if not self.gui:
            return
        cam = p.getDebugVisualizerCamera(physicsClientId=self.sim.client)
        width, height = int(cam[0]), int(cam[1])
        if width <= 0 or height <= 0:  # Minimized/uninitialized viewport.
            return
        view = np.asarray(cam[2], dtype=float).reshape(4, 4, order="F")
        projection = np.asarray(cam[3], dtype=float).reshape(4, 4, order="F")
        try:
            screen_to_world = np.linalg.inv(projection @ view)
        except np.linalg.LinAlgError:
            return  # Camera has not supplied a usable projection yet.
        terms = self.reward_metrics
        def value(key):
            return f"{terms.get(key, 0.):+.4f}"
        rows = [
            f"{self.wall.name.upper()}  step {self.steps}  {self.reason}",
            f"STEP REWARD {reward:+.4f}   EPISODE RETURN {self.episode_return:+.3f}",
            f"Coverage: target {info['target_coverage']:.1%} / whole wall {info['wall_coverage']:.1%}",
            f"Contact: {info['contact']}   gap {info['gap_mm']:.1f} mm",
            f"New paint {value('new_paint')}   orientation {value('orientation')}   approach {value('approach')}",
            f"Gap {value('wall_gap')}   lost contact {value('lost_contact')}   no new paint {value('no_new_paint')}",
            f"Repaint {value('repaint')}   rotation {value('rotation_effort')}   action change {value('action_change')}",
            f"Time {value('time')}   rejected {value('rejected')}   failure {value('failure')}   success {value('success')}",
        ]
        while len(self.hud_ids) < len(rows):
            self.hud_ids.append(-1)
        for index, text in enumerate(rows):
            # Bitmap text stays camera-facing when textOrientation is omitted.
            point = screen_to_world @ np.array([
                -1. + 24./width, 1. - 2.*(28. + 23.*index)/height, 0., 1.])
            if abs(point[3]) < 1e-12:
                continue
            color = ([.05, .5, .12] if reward >= 0 else [.85, .12, .08]) if index == 1 else [.08, .08, .08]
            self.hud_ids[index] = p.addUserDebugText(
                text, (point[:3]/point[3]).tolist(), textColorRGB=color,
                textSize=1.2 if index == 1 else 1., lifeTime=0,
                replaceItemUniqueId=self.hud_ids[index], physicsClientId=self.sim.client)

    def _info(self):
        info = {"wall": self.wall.name, "steps": self.steps, "reason": self.reason,
                "is_success": self.reason == "success", "contact": self.contact,
                "target_coverage": self.area/self.target_area,
                "wall_coverage": self.wall.canvas.painted_area/self.wall.canvas.paintable_area,
                "target_wall_fraction": self.target_area/self.wall.canvas.paintable_area,
                "target_area_m2": self.target_area,
                "paintable_wall_m2": self.wall.canvas.paintable_area,
                "no_new_paint_seconds": (self.steps-self.last_progress_step)*DT,
                "idle_contact_seconds": self.idle_steps*DT,
                "command_status": self.command_status,
                "rejected_fraction": self.rejected_steps/max(1, self.steps),
                "recovery_fraction": self.recovery_steps/max(1, self.steps),
                "servo_scale": self.servo_scale,
                "tracking_error_mm": float(np.linalg.norm(self._pose()[0]-self.command[:3])*1000),
                "joint_target_error": self.joint_target_error.tolist(),
                "action": self.previous_action.tolist()}
        info.update(self._contact_state())
        return info

    def close(self):
        self.done = True
        scene, self.sim = self.sim, None
        client, self._client = self._client, None
        self.robot = self.planner = self.wall = None
        try:
            if client is not None and p.isConnected(client):
                p.disconnect(client)
        finally:
            if scene is not None:
                scene.close()


class CoverageLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.recent = deque(maxlen=50)

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                self.recent.append(info)
        if self.num_timesteps % 1000 == 0:
            for key in ("target_coverage", "wall_coverage", "target_wall_fraction"):
                if self.recent:
                    self.logger.record("coverage/"+key, np.mean([i[key] for i in self.recent]))
            infos = self.locals.get("infos", [])
            if infos:
                for key in ("gap_mm", "axis_tilt_deg", "tracking_error_mm", "rejected_fraction", "recovery_fraction"):
                    self.logger.record("controller/"+key, infos[0][key])
                for key, value in infos[0].get("reward_terms", {}).items():
                    self.logger.record("reward/"+key, value)
            self.logger.dump(self.num_timesteps)
        return True


def preflight(config, gui=False):
    """Check Gym API, full-wall kinematics and actual contact in both orientations."""
    cfg = Config(**asdict(config))
    cfg.layout, cfg.randomize_size = "solid", False
    cfg.walls = WALLS
    print("[check] Gym API checks (headless)", flush=True)
    api_env = WholeWallEnv(cfg, gui=False)
    try:
        check_env(api_env, warn=True, skip_render_check=True)
        print("[check] Gym API passed", flush=True)
    finally:
        api_env.close()
    env = WholeWallEnv(cfg, gui)
    try:
        for name in WALLS:
            print(f"[check] {name}: reset and validate robot", flush=True)
            env.reset(seed=123, options={"wall": name})
            print(f"[check] {name}: geometry probes, client={env._client}, "
                  f"body={env.robot.id}, roller_link={env.robot.roller_link_index}", flush=True)
            checked = 0
            # Actual Bullet FK and collision probes, only during this diagnostic.
            for theta in (0., math.pi/4, math.pi/2):
                u0, u1, v0, v1 = center_bounds(env.wall.length, env.wall.height, theta)
                for u in np.linspace(u0+.001, u1-.001, 5):
                    for v in np.linspace(v0+.001, v1-.001, 5):
                        command = np.array([u, v, .0595, theta])
                        q = env._ik(command)
                        if q is None:
                            continue
                        env.robot.set_joint_targets(q, instant=True)
                        p.performCollisionDetection()
                        env.verify_pose(command)
                        if env.planner.current_collision(env.wall.body_id, executing=True):
                            raise RuntimeError(f"Geometry probe collided on {name}, pose {command}")
                        checked += 1
            for theta, action in ((0., [0, .6, -.2, 0]), (math.pi/2, [.6, 0, -.2, 0])):
                print(f"[check] {name}: {'vertical' if theta == 0 else 'horizontal'} painting stroke", flush=True)
                env.reset(seed=123, options={"wall": name,
                    "command": [env.wall.length/2, 1., .0595, theta]})
                for _ in range(100):
                    obs, reward, terminated, truncated, info = env.step(np.array(action))
                    if not env.observation_space.contains(obs) or not math.isfinite(reward):
                        raise RuntimeError("Invalid observation/reward")
                    if terminated or truncated:
                        raise RuntimeError(f"Short diagnostic stroke ended early: {info}")
                if env.area < .005:
                    raise RuntimeError(f"No meaningful paint from contact on {name}, theta={theta}: {json.dumps(info)}")
            # Exercise a random pose in the same region used for training resets.
            print(f"[check] {name}: random-start stroke", flush=True)
            env.reset(seed=917, options={"wall": name})
            theta = env.command[3]
            action = np.array([-.5*math.sin(theta), .5*math.cos(theta), -.2, 0.])
            for _ in range(100):
                _, _, term, trunc, random_info = env.step(action)
                if term or trunc:
                    raise RuntimeError("Random-start contact check failed: "+json.dumps(random_info))
            if env.area < .005:
                raise RuntimeError("Random-start stroke did not paint: "+json.dumps(random_info))
            print(f"PASS {name}: {checked} geometry probes and vertical/horizontal contact strokes; "
                  f"random-start area={env.area:.4f}m2 gap={random_info['gap_mm']:.2f}mm; "
                  f"target includes {info['target_wall_fraction']:.1%} of paintable wall")
        print("Preflight passed. This checks geometry/API/contact, not learned coverage performance.", flush=True)
    finally:
        env.close()


def checkpoint_path(value):
    path = Path(value).expanduser()
    if not path.is_file() and path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train first, or select an existing .zip.")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--check-rewards", action="store_true", help="quick reward checks without opening a simulation")
    mode.add_argument("--evaluate", metavar="MODEL.zip")
    mode.add_argument("--resume", metavar="MODEL.zip")
    mode.add_argument("--warm-start", metavar="MODEL.zip", help="copy actor only; new critics and replay")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="detailed evaluation output every 50 steps")
    parser.add_argument("--steps", type=int, default=300000, help="additional environment steps to train")
    parser.add_argument("--out", default="runs/td3_wall_v3")
    parser.add_argument("--walls", nargs="+", choices=WALLS)
    parser.add_argument("--layout", choices=("solid", "original", "random"))
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--randomize-size", dest="randomize_size", action="store_true", default=None)
    sizes.add_argument("--fixed-size", dest="randomize_size", action="store_false")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--stall-seconds", type=float)
    parser.add_argument("--idle-contact-seconds", type=float)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="CPU is a reliable starting choice on Mac")
    parser.add_argument("--buffer-size", type=int, default=20000, help="about 0.4 GB observations at default")
    parser.add_argument("--replay-buffer", help="load matching replay .pkl when resuming")
    parser.add_argument("--save-replay", action="store_true", help="also save replay_buffer.pkl at the end")
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0 or args.buffer_size < 256:
        parser.error("steps/episodes must be positive and buffer-size must be >=256")
    if args.replay_buffer and not args.resume:
        parser.error("--replay-buffer requires --resume")
    source_model = args.evaluate or args.resume or args.warm_start
    model_path = checkpoint_path(source_model) if source_model else None
    cfg = Config()
    saved_cfg = None
    if model_path:
        config_path = model_path.parent / "config.json"
        if not config_path.is_file() and model_path.parent.name == "checkpoints":
            config_path = model_path.parent.parent / "config.json"
        if not config_path.is_file():
            parser.error(f"Missing {config_path}; keep this script's config beside the model")
        saved = json.loads(config_path.read_text())
        source_version = saved.get("version")
        if source_version not in (VERSION,) + LEGACY_VERSIONS:
            parser.error("This is not a compatible whole-wall checkpoint. Start a new run.")
        if source_version in LEGACY_VERSIONS:
            if args.resume:
                parser.error("v3 rewards changed: start a fresh run in a new --out directory. Old critics/replay are incompatible.")
            print(f"Using {source_version} actor with v3 rewards and the corrected controller. "
                  "Returns are not comparable to the old evaluation. Changing rewards does not retrain this actor.")
        saved_cfg = saved["environment"]
        cfg = Config(**saved_cfg)
    for item in fields(Config):
        value = getattr(args, item.name, None)
        if value is not None:
            setattr(cfg, item.name, tuple(value) if item.name == "walls" else value)
    cfg.validate()
    if args.check_rewards:
        check_rewards()
        return
    if args.check:
        check_rewards()
        preflight(cfg, args.gui)
        return
    if args.evaluate:
        env = WholeWallEnv(cfg, args.gui)
        try:
            model = TD3.load(str(model_path), device=args.device, env=env)
            results = []
            for episode in range(args.episodes):
                name = cfg.walls[episode % len(cfg.walls)]
                obs, info = env.reset(seed=args.seed+10000+episode, options={"wall": name})
                total = 0.
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    if env.steps % (50 if args.diagnostics else 200) == 0:
                        print(f"{name} step={env.steps} target={info['target_coverage']:.1%} "
                              f"whole_wall={info['wall_coverage']:.1%} contact={info['contact']} "
                              f"gap={info['gap_mm']:.2f}mm tilt={info['axis_tilt_deg']:.1f}deg "
                              f"controller={info['command_status']} return={total:.2f}")
                        if args.diagnostics:
                            print(json.dumps({k: info[k] for k in ("raw_contacts", "contact_reason",
                                "endpoint_gaps_mm", "tracking_error_mm", "joint_target_error", "servo_scale",
                                "rejected_fraction", "recovery_fraction", "paint_contact_fraction", "action", "reward_terms")}))
                    if terminated or truncated:
                        break
                result = {k: v for k, v in info.items() if k not in ("reward_terms", "rejected_command")}
                result.update(episode=episode+1, episode_return=round(total, 3))
                results.append(result)
                print(json.dumps(result, indent=2))
            print(f"Mean whole-wall coverage: {np.mean([r['wall_coverage'] for r in results]):.1%}; "
                  f"target success rate: {np.mean([r['is_success'] for r in results]):.1%}")
        finally:
            env.close()
        return
    out = Path(args.out)
    if args.warm_start and model_path.parent.resolve() == out.resolve():
        parser.error("Use a new --out directory for --warm-start to preserve the previous checkpoint")
    if (out / "final.zip").exists() and not args.resume:
        parser.error("Output contains final.zip; use --resume or a new --out directory")
    if args.replay_buffer and saved_cfg != json.loads(json.dumps(asdict(cfg))):
        parser.error("Do not reuse replay when environment/reward configuration changes")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"version": VERSION, "implementation": "v3.1-live-reward-display", "gamma": GAMMA, "environment": asdict(cfg),
        "seed": args.seed, "source_checkpoint": str(model_path) if model_path else None}, indent=2)+"\n")
    raw = WholeWallEnv(cfg, args.gui)
    env = Monitor(raw, str(out / "monitor.csv"), info_keywords=(
        "target_coverage", "wall_coverage", "target_wall_fraction", "reason", "wall"))
    noise = NormalActionNoise(np.zeros(4), .12*np.ones(4))
    model = None
    try:
        if args.resume:
            model = TD3.load(str(model_path), env=env, device=args.device, buffer_size=args.buffer_size,
                             action_noise=noise, seed=args.seed)
            if not math.isclose(model.gamma, GAMMA, rel_tol=0., abs_tol=1e-12):
                raise ValueError("Checkpoint discount differs from v3 potential shaping; start a fresh run")
            if args.replay_buffer:
                model.load_replay_buffer(args.replay_buffer)
            else:
                # Absolute SB3 timestep threshold: collect warm-up after existing timesteps.
                model.learning_starts = model.num_timesteps + 2000
                print("Resuming weights and optimizers with a fresh replay buffer (2,000 warm-up steps).")
        else:
            model = TD3("MultiInputPolicy", env, learning_rate=3e-4, buffer_size=args.buffer_size,
                        learning_starts=2000, batch_size=128, tau=.005, gamma=GAMMA,
                        train_freq=1, gradient_steps=1, policy_delay=2,
                        target_policy_noise=.2, target_noise_clip=.5, action_noise=noise,
                        policy_kwargs={"net_arch": [256, 256]}, seed=args.seed,
                        device=args.device, verbose=1)
            if args.warm_start:
                previous = TD3.load(str(model_path), device=args.device)
                if previous.action_space != raw.action_space or previous.observation_space != raw.observation_space:
                    raise ValueError("Warm-start requires this file's four-action whole-wall policy")
                model.actor.load_state_dict(previous.actor.state_dict())
                model.actor_target.load_state_dict(model.actor.state_dict())
                del previous
                print("Copied actor weights only; critics, optimizers and replay are new for reward v3. "
                      "A collapsed actor can retain its bad behavior; fresh training is recommended.")
        model.set_logger(configure(str(out), ["stdout", "csv"]))
        checkpoints = CheckpointCallback(save_freq=10000, save_path=str(out / "checkpoints"), name_prefix="td3_wall")
        print(f"Training {args.steps:,} additional steps; episode cutoff {cfg.max_episode_steps:,}. "
              "One selected wall per episode; unlimited paint. Watch whole_wall coverage.")
        try:
            model.learn(total_timesteps=args.steps, callback=[checkpoints, CoverageLog()],
                        log_interval=5, reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            print("Training interrupted; saving the current model.")
        model.save(str(out / "final.zip"))
        if args.save_replay:
            model.save_replay_buffer(str(out / "replay_buffer.pkl"))
        print(f"Saved {out / 'final.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
