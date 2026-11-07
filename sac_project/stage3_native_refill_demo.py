#!/usr/bin/env python3
"""Stage 3 V4-native painting with in-world Stage 4 V2 dipping.

Stage3TrainEnv owns observations, V8 planner, paint masks, action scaling and
termination. No replacement painting observation or geometric painting policy.
Navigation to the can is explicitly geometric; dipping uses the frozen 36D actor.
"""
import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pybullet as p

from train_stage3_goal_dagger_v4 import Stage3TrainEnv
from dip_v2_environment import local_ik, contained, ACTION_SCALE, CONTROL_DT
from simulation.paint_can import PaintCan
from stable_baselines3 import SAC


def safe_json(obj):
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(type(obj).__name__)


class Refill:
    def __init__(self, env, actor, can_u=2.65, can_n=2.4):
        self.env, self.actor = env, actor
        self.wall = env.wall
        self.refills = 0
        self.actions = 0
        self.events = []
        # The robot's fixed base is mounted at wall.origin with wall.yaw.
        # Place the can in this same wall-local frame, inside the room.
        origin = self.wall.origin
        world = origin + self.wall.u_dir * can_u + self.wall.normal * can_n
        self.can = PaintCan(position=(float(world[0]), float(world[1]), 0.))
        self.last = np.zeros(4, np.float32)

    def log(self, event, **kwargs):
        row = dict(event=event, physics_steps=self.actions * 12, **kwargs)
        self.events.append(row)
        print('[refill] ' + json.dumps(row, default=safe_json), flush=True)

    def world_to_local(self, world):
        point = np.asarray(world, float)
        offset = point - self.wall.origin
        return np.array([float(offset @ self.wall.u_dir), float(offset @ self.wall.normal), float(point[2])])

    def local_ik_world(self, xyz, tilt=0.):
        local = self.world_to_local(xyz)
        q = local_ik(local, tilt)
        if q is None: raise RuntimeError('Dip IK failed')
        q = np.asarray(q, float)
        lo, hi = np.asarray(self.env.robot.lower_limits), np.asarray(self.env.robot.upper_limits)
        if np.any(q < lo) or np.any(q > hi):
            raise RuntimeError('Dip IK outside Stage3 robot joint limits')
        return q

    def step_physics(self, release=False):
        env = self.env
        for _ in range(12):
            env.sim.step()
            # The roller is allowed to touch the can during dipping; the
            # robot chassis/arm may not penetrate a room wall.
            for wall in env.sim.walls:
                for hit in (p.getContactPoints(env.robot.id, wall.body_id) or ()):
                    if hit[8] < -.002 and not (release and wall.body_id == env.wall.body_id and hit[3] == env.robot.roller_link_index):
                        raise RuntimeError(f'Collision with {wall.name}: link={hit[3]} depth={hit[8]:.5f}')
        self.actions += 1

    def move(self, q, release=False):
        env = self.env
        q = np.asarray(q, float)
        start = np.asarray(env.robot.get_joint_states(), float)
        inc = np.array([.03, .03, .07, .06, .025, .08, .08])
        n = int(max(1, np.ceil(np.max(np.abs(q - start) / inc))))
        for i in range(1, n + 1):
            target = start + (q - start) * i / n
            # Physical position-control only, never resetJointState.
            for _ in range(18):
                env.robot.set_joint_targets(target)
                self.step_physics(release=release)
                actual = np.asarray(env.robot.get_joint_states(), float)
                if np.max(np.abs(actual - target)) < .025: break
            else:
                raise RuntimeError(f'Refill travel motor tracking timeout, max_error={np.max(np.abs(actual-target)):.4f}')

    def run(self):
        env = self.env
        if env.done: raise RuntimeError('Stage3 already terminated before refill')
        saved = np.asarray(env.command, float).copy()
        saved_q = np.asarray(env.robot.get_joint_states(), float).copy()
        self.log('pause', wall=env.wall.name, command=saved.tolist())
        env.wall.canvas.lift()
        # Native Stage3 IK: retract at same u/v, then safely stage arm.
        retract = saved.copy(); retract[2] = .13
        q_retract = env._ik(retract, previous=saved_q)
        if q_retract is None: raise RuntimeError('Native Stage3 retraction IK unavailable')
        self.move(q_retract, release=True)
        approach = np.asarray(self.can.get_approach_position(env.robot.ROLLER_LENGTH, env.robot.ROLLER_RADIUS, .05), float)
        inside = np.asarray(self.can.get_dip_position(), float)
        stage = approach.copy(); stage[2] = max(stage[2] + .12, .55)
        self.move(self.local_ik_world(stage))
        self.move(self.local_ik_world(approach))
        self.log('at_can')
        phase, dwell, refilled = 0, 0., False
        self.last = np.zeros(4, np.float32)
        for k in range(800):
            a, b = map(lambda x: np.asarray(x, float), env.robot.get_roller_axis_segment())
            center = (a+b)/2
            axis = (a-b)/env.robot.ROLLER_LENGTH
            tilt = math.atan2(float(axis[0]), float(axis[2]))
            goal = approach if phase in (0, 3) else inside
            q = np.asarray(env.robot.get_joint_states(), float)
            dq = np.asarray([s[1] for s in p.getJointStates(env.robot.id, env.robot.joint_indices)], float)
            obs = np.concatenate(((center-np.asarray(self.can.position))/.2, axis,
                (goal-center)/.2, q, dq, self.last, np.eye(4)[phase],
                [dwell/.5], [self.can.inner_radius, self.can.depth, self.can.wall_thickness],
                [float(refilled)])).astype(np.float32)
            if obs.shape != (36,): raise RuntimeError('Dip observation must be 36D')
            action = np.asarray(self.actor.predict(obs, deterministic=True)[0], float)
            if action.shape != (4,) or not np.isfinite(action).all():
                raise RuntimeError('Invalid learned dip action')
            self.last = action.astype(np.float32)
            target = self.local_ik_world(center + np.clip(action[:3], -1, 1) * ACTION_SCALE[:3],
                                         tilt + float(np.clip(action[3], -1, 1)) * ACTION_SCALE[3])
            env.robot.set_joint_targets(target)
            held = phase == 2 and contained(a,b,env.robot.ROLLER_RADIUS,self.can.position,
                        self.can.inner_radius,self.can.depth,self.can.wall_thickness)
            self.step_physics()
            a,b = map(lambda x: np.asarray(x,float),env.robot.get_roller_axis_segment())
            center = (a+b)/2
            axis = (a-b)/env.robot.ROLLER_LENGTH
            tilt = math.atan2(float(axis[0]),float(axis[2]))
            near = np.linalg.norm(center-goal) < .008 and abs(tilt) < .05
            if phase == 0 and near: phase=1
            elif phase == 1 and near and contained(a,b,env.robot.ROLLER_RADIUS,self.can.position,
                         self.can.inner_radius,self.can.depth,self.can.wall_thickness): phase=2; dwell=0.
            elif phase == 2:
                dwell = dwell + CONTROL_DT if held else 0.
                if dwell >= .5-1e-9:
                    refilled=True; phase=3; self.log('refill_verified',dwell_seconds=round(dwell,3))
            elif phase == 3 and near:
                if not refilled: raise RuntimeError('Dip completed without refill')
                break
        else:
            raise RuntimeError('Neural dipping timed out')
        self.move(self.local_ik_world(stage))
        self.move(q_retract)
        self.move(saved_q, release=True)
        # The Stage3 planner, command and previous action remain intact.
        # Recompute cached planner features after physical repositioning.
        env.wall.canvas.lift()
        env.last_progress_step = env.steps
        env.idle_steps = 0
        env.meaningful_area = 0.
        env._prepare()
        self.refills += 1
        self.log('resumed', refills=self.refills)
        return env._obs()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paint-actor',type=Path,required=True)
    ap.add_argument('--dip-actor',type=Path,required=True)
    ap.add_argument('--wall',choices=['south','east','north','west'],default='south')
    ap.add_argument('--seed',type=int,default=860000)
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--max-actions',type=int,default=30000)
    ap.add_argument('--refill-capacity-m2',type=float,default=.15)
    ap.add_argument('--low-fraction',type=float,default=.15)
    ap.add_argument('--max-refills',type=int,default=20)
    ap.add_argument('--out',type=Path,default=Path('sac_project/runs/stage3_native_refill'))
    args=ap.parse_args()
    for ckpt in (args.paint_actor,args.dip_actor):
        if not ckpt.is_file(): ap.error(f'Missing checkpoint: {ckpt}')
    args.out.mkdir(parents=True,exist_ok=True)
    result={'result':'not_started'}
    env=None; refill=None
    try:
        env=Stage3TrainEnv([args.wall],episode_steps=args.max_actions)
        env.gui=args.gui
        env.render_mode='human' if args.gui else None
        obs,info=env.reset(seed=args.seed,options={'wall':args.wall})
        painter=SAC.load(str(args.paint_actor),device='cpu')
        dipper=SAC.load(str(args.dip_actor),device='cpu')
        if painter.observation_space['state'].shape != (56,) or painter.action_space.shape != (4,):
            raise RuntimeError('Painting checkpoint incompatible with Stage3 V4 56D')
        if dipper.observation_space.shape != (36,) or dipper.action_space.shape != (4,):
            raise RuntimeError('Dipping checkpoint incompatible with 36D dipping')
        refill=Refill(env,dipper)
        paint_level=1.
        last_area=float(env.area)
        reason='action_limit'
        for k in range(args.max_actions):
            if paint_level <= args.low_fraction:
                if refill.refills >= args.max_refills:
                    reason='refill_limit';break
                obs=refill.run()
                paint_level=1.
                last_area=float(env.area)
            action,_=painter.predict(obs,deterministic=True)
            obs,_,terminated,truncated,info=env.step(action)
            new_area=max(0.,float(env.area)-last_area)
            paint_level=max(0.,paint_level-new_area/args.refill_capacity_m2)
            last_area=float(env.area)
            if (k+1)%200==0:
                print('[stage3] '+json.dumps(dict(actions=k+1,coverage=info['target_coverage'],
                     paint_level=paint_level,refills=refill.refills,reason=info['reason'])),flush=True)
            if terminated or truncated:
                reason=info['reason'];break
        result=dict(result=reason,wall=args.wall,seed=args.seed,coverage=float(info['target_coverage']),
                    whole_wall_coverage=float(info['wall_coverage']),painting_actions=k+1,
                    refills=refill.refills,neural_paint=True,neural_dip=True,
                    geometric_refill_travel=True,observation='original_Stage3TrainEnv_56D',
                    events=refill.events)
    except Exception as exc:
        result=dict(result='failed',error=str(exc),traceback=traceback.format_exc(),
                    wall=args.wall,refills=refill.refills if refill else 0)
        print('[failure] '+str(exc),flush=True)
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        (args.out/'result.json').write_text(json.dumps(result,indent=2,default=safe_json))
        print('[summary] '+json.dumps({k:v for k,v in result.items() if k not in ('events','traceback')},default=safe_json),flush=True)

if __name__=='__main__': main()
