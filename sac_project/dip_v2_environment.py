#!/usr/bin/env python3
"""Local paint-dipping SAC experiment for the archived telescopic Robot.

Run from project root: python sac_project/train_stage4_dip_v1.py --help
Requires simulation/robot.py and simulation/paint_can.py from the project.
This is a separate local skill: no wall departure, navigation, or painter handoff.
The network chooses all four Cartesian/tilt increments; analytic IK maps them
to motors. A geometric phase machine chooses goals, never student actions.
No existing main/cleanup weights are loaded or modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

PHASES = ("align", "lower", "hold", "withdraw")
CONTROL_DT = .05
ACTION_SCALE = np.array([.005, .005, .004, .04], dtype=np.float64)
OBS_DIM = 36


def contained(a, b, radius, position, inner_radius, depth, thickness):
    """Same conservative full-barrel geometry as archived PaintCan.try_dip.

    Pure query: unlike try_dip(), this does not mark the can as dipped before
    the required uninterrupted dwell interval has completed.
    """
    a, b, c = map(lambda x: np.asarray(x, dtype=float), (a, b, position))
    length = np.linalg.norm(a-b)
    if length <= 1e-12 or radius <= 0:
        return False
    axis_z = min(1., abs(a[2]-b[2])/length)
    radial_z = radius * math.sqrt(max(0., 1.-axis_z**2))
    return bool(all(np.linalg.norm(v[:2]-c[:2])+radius <= inner_radius-.002
                    and v[2]-radial_z >= c[2]+thickness+.002
                    and v[2]+radial_z <= c[2]+depth-.002 for v in (a, b)))


def local_ik(xyz, tilt):
    """Exact kinematics for the archived robot, fixed base yaw and wrist roll.

    0.86 m fixed reach, 1.0 m shoulder height, 0.02 m shoulder offset.
    Keep 0.85 m horizontal arm reach, letting the base follow tool XY.
    Reset validates this mapping against actual forward kinematics.
    """
    x, y, z = xyz
    pitch = math.atan2(1.-z, .85)
    extension = math.hypot(.85, 1.-z)-.86
    return np.array([x-.87, y, 0., pitch, extension, 0., tilt-pitch])


def make_env(project_root, gui=False, max_steps=800, spread=.08):
    import gymnasium as gym
    import pybullet as p
    sys.path.insert(0, str(Path(project_root).resolve()))
    from simulation.robot import Robot
    from simulation.paint_can import PaintCan

    class DipEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
            # Archived project classes use PyBullet's default client. Keep one
            # environment per process; concurrent envs require subprocesses.
            if p.isConnected():
                raise RuntimeError("Use one DipEnv per process (archived PyBullet API).")
            self.client = p.connect(p.GUI if gui else p.DIRECT)

        def pose(self):
            a, b = self.robot.get_roller_axis_segment()
            a, b = np.asarray(a), np.asarray(b)
            axis = (a-b)/self.robot.ROLLER_LENGTH
            return (a+b)/2., axis, math.atan2(axis[0], axis[2])

        def goal(self):
            return self.above if self.phase in (0, 3) else self.inside

        def inside_can(self):
            a, b = self.robot.get_roller_axis_segment()
            return contained(a, b, self.robot.ROLLER_RADIUS, self.can.position,
                             self.can.inner_radius, self.can.depth, self.can.wall_thickness)

        def collision(self):
            for body in (self.floor, *self.can.collision_body_ids):
                points = p.getClosestPoints(self.robot.id, body, distance=0.)
                if points is None:
                    raise RuntimeError("Collision query unavailable")
                for contact in points:
                    tolerance = .0005 if contact[3] == self.robot.roller_link_index else .002
                    if contact[8] < -tolerance:
                        return True
            return False

        def obs(self):
            center, axis, _ = self.pose()
            states = p.getJointStates(self.robot.id, self.robot.joint_indices)
            q = np.array([s[0] for s in states])
            qd = np.array([s[1] for s in states])
            value = np.concatenate(((center-np.asarray(self.can.position))/.2, axis,
                                    (self.goal()-center)/.2, q, qd, self.last_action,
                                    np.eye(4)[self.phase], [self.dwell/.5],
                                    [self.can.inner_radius, self.can.depth, self.can.wall_thickness],
                                    [float(self.refilled)])).astype(np.float32)
            if value.shape != (OBS_DIM,) or not np.isfinite(value).all():
                raise RuntimeError("Invalid dip observation")
            return value

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1/240)
            floor_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3., 3., .05])
            self.floor = p.createMultiBody(0, floor_shape, basePosition=[1.5, 1.5, -.05])
            can_xy = np.array([1.5, 1.5])+self.np_random.uniform(-.15, .15, 2)
            self.can = PaintCan(position=(*can_xy, 0.))
            self.robot = Robot(start_xy=(can_xy[0]-.87, can_xy[1]))
            self.above = np.asarray(self.can.get_approach_position(
                self.robot.ROLLER_LENGTH, self.robot.ROLLER_RADIUS, .05), dtype=float)
            self.inside = np.asarray(self.can.get_dip_position(), dtype=float)
            initial = self.above + np.r_[self.np_random.uniform(-spread, spread, 2),
                                        self.np_random.uniform(.01, .05)]
            tilt = self.np_random.uniform(-.15, .15)
            q = local_ik(initial, tilt)
            self.robot.set_joint_targets(q, instant=True)  # reset only
            p.performCollisionDetection()
            center, _, sensed_tilt = self.pose()
            if np.linalg.norm(center-initial) > .001 or abs(sensed_tilt-tilt) > .01:
                raise RuntimeError("Robot geometry changed: adapt local_ik before training")
            if self.collision():
                raise RuntimeError("Reset in collision")
            self.phase = 0
            self.dwell = 0.
            self.refilled = False
            self.steps = 0
            self.last_action = np.zeros(4)
            self.phase_steps = [0]*4
            self.ended = False
            return self.obs(), {}

        def teacher(self):
            # Only explicitly requested expert rollouts call this function.
            center, _, tilt = self.pose()
            return np.clip(np.r_[.4*(self.goal()-center), -.4*tilt]/ACTION_SCALE,
                           -1., 1.).astype(np.float32)

        def step(self, action):
            if self.ended:
                raise RuntimeError("Call reset after episode end")
            action = np.asarray(action, dtype=float)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError("Action must contain four finite values")
            action = np.clip(action, -1., 1.)
            self.last_action = action.copy()
            self.steps += 1
            self.phase_steps[self.phase] += 1
            center, _, tilt = self.pose()
            goal = self.goal().copy()
            old_error = np.linalg.norm(center-goal)/.1 + abs(tilt)/.3
            proposed = center+action[:3]*ACTION_SCALE[:3]
            q = local_ik(proposed, tilt+action[3]*ACTION_SCALE[3])
            invalid = not all(lo <= v <= hi for v, lo, hi in
                              zip(q, self.robot.lower_limits, self.robot.upper_limits))
            collided = False
            held_inside = self.phase == 2 and self.inside_can()
            if not invalid:
                self.robot.set_joint_targets(q)  # physical motors, no pose reset
                for _ in range(12):
                    p.stepSimulation()
                    if self.phase == 2:
                        held_inside = held_inside and self.inside_can()
                    if self.collision():
                        collided = True
                        break
            center, axis, tilt = self.pose()
            new_error = np.linalg.norm(center-goal)/.1 + abs(tilt)/.3
            reward = 5.*(old_error-new_error)-.02-.002*float(np.square(action).sum())
            success = False
            reason = "running"
            if invalid or collided:
                reward -= 20.
                reason = "joint_limit" if invalid else "collision"
            else:
                near = np.linalg.norm(center-goal) < .008 and abs(tilt) < .05
                if self.phase == 0 and near:
                    self.phase = 1
                    reward += 2.
                elif self.phase == 1 and near and self.inside_can():
                    self.phase = 2
                    self.dwell = 0.
                    reward += 2.
                elif self.phase == 2:
                    # Continuous physical containment, not just a waypoint hit.
                    self.dwell = self.dwell+CONTROL_DT if held_inside else 0.
                    if self.dwell >= .5-1e-9:
                        self.refilled = True
                        self.can.dipped = True
                        self.robot.set_roller_color([0., 0., 1., 1.])
                        self.phase = 3
                        reward += 5.
                elif self.phase == 3:
                    lower = (center[2]-.5*self.robot.ROLLER_LENGTH*abs(axis[2])
                             -self.robot.ROLLER_RADIUS*math.sqrt(max(0., 1.-axis[2]**2)))
                    if near and lower > self.can.position[2]+self.can.depth+.02:
                        success = self.refilled
                        reward += 20.
                        reason = "success"
            terminated = bool(invalid or collided or success)
            truncated = bool(self.steps >= max_steps and not terminated)
            if truncated:
                reason = "timeout"
            self.ended = terminated or truncated
            info = dict(is_success=success, collision=collided, reason=reason,
                        phase=PHASES[self.phase], phase_steps=dict(zip(PHASES, self.phase_steps)),
                        dwell_seconds=self.dwell, refilled=self.refilled,
                        steps=self.steps, target_error_m=float(np.linalg.norm(center-self.goal())),
                        tilt_error_rad=float(abs(tilt)))
            return self.obs(), float(reward), terminated, truncated, info

        def close(self):
            if p.isConnected(self.client):
                p.disconnect(self.client)

    return DipEnv()


def evaluate(env, model, episodes, seed, expert=False):
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed+i)
        total_reward = 0.
        while True:
            action = env.teacher() if expert else model.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        row = dict(seed=seed+i, controller="expert" if expert else "student",
                   episode_reward=total_reward, **info)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(controller="expert" if expert else "student", episodes=rows,
                success_rate=float(np.mean([r["is_success"] for r in rows])),
                collision_rate=float(np.mean([r["collision"] for r in rows])))


def self_test():
    # Dwell timing, containment edge cases and nonteleporting step behavior are
    # additionally checked with --mode expert and --mode smoke in real physics.
    kw = dict(radius=.06, position=[0, 0, 0], inner_radius=.1, depth=.28, thickness=.015)
    assert contained([0, 0, .2375], [0, 0, .0575], **kw)
    assert not contained([.04, 0, .2375], [.04, 0, .0575], **kw)
    assert not contained([0, 0, .29], [0, 0, .11], **kw)
    assert not contained([.09, 0, .1475], [-.09, 0, .1475], **kw)
    assert not contained([0, 0, .19], [0, 0, .01], **kw)
    assert not contained([0, 0, .15], [0, 0, .15], **kw)
    assert np.isfinite(local_ik([1.5, 1.5, .1475], 0.)).all()
    print("Geometry self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--mode", choices=("train", "evaluate", "expert", "smoke"), default="train")
    ap.add_argument("--out", type=Path, default=Path("sac_project/runs/stage4_dip_v1"))
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--episode-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=5478)
    ap.add_argument("--eval-seed", type=int, default=950000)
    ap.add_argument("--spread", type=float, default=.08)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.episodes, args.episode_steps, args.steps) < 1 or not 0 <= args.spread <= .15:
        ap.error("Positive counts required; spread must be between 0 and 0.15 m")
    for f in ("simulation/robot.py", "simulation/paint_can.py"):
        if not (args.project_root/f).is_file():
            ap.error(f"Missing {args.project_root/f}; pass --project-root")
    if args.mode == "evaluate" and not args.checkpoint:
        ap.error("--checkpoint is required for student evaluation")
    if args.mode == "train" and args.checkpoint:
        ap.error("This v1 starts a new actor; resume training is not implemented")
    # Never overwrite a prior run. Use a separate output directory per command.
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    env = make_env(args.project_root, args.gui, args.episode_steps, args.spread)
    try:
        if args.mode == "smoke":
            from gymnasium.utils.env_checker import check_env
            check_env(env, skip_render_check=True)
            report = evaluate(env, None, args.episodes, args.eval_seed, expert=True)
        elif args.mode == "expert":
            report = evaluate(env, None, args.episodes, args.eval_seed, expert=True)
        else:
            import torch
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import CheckpointCallback
            from stable_baselines3.common.monitor import Monitor
            torch.set_num_threads(1)
            if args.mode == "train":
                preflight = evaluate(env, None, 4, args.seed+10000, expert=True)
                (args.out/"expert_preflight.json").write_text(json.dumps(preflight, indent=2))
                if preflight['success_rate'] < 1.:
                    raise RuntimeError("Expert preflight failed; inspect geometry before training")
                env = Monitor(env, filename=str(args.out/"monitor.csv"))
                env.reset(seed=args.seed)
                model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=200000,
                            learning_starts=2000, batch_size=256, gamma=.99, tau=.005,
                            policy_kwargs=dict(net_arch=[128, 128]), device="cpu",
                            seed=args.seed, verbose=1)
                model.learn(args.steps, callback=CheckpointCallback(
                    save_freq=25000, save_path=str(args.out/"checkpoints"), name_prefix="dip"))
                model.save(args.out/"final_actor")
            else:
                model = SAC.load(args.checkpoint, device="cpu")
                if model.observation_space != env.observation_space or model.action_space != env.action_space:
                    raise ValueError("Checkpoint has incompatible observation/action spaces")
            report = evaluate(env, model, args.episodes, args.eval_seed)
        (args.out/"evaluation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
