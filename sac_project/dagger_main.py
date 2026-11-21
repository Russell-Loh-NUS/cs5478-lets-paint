#!/usr/bin/env python3
"""Original Stage 3 painting loop with persistent-room travel and refill/resume.

Requires the project's simulation/, planning/ and sac_project/ modules.
All walls share one PyBullet world and one robot. No reset between walls.
"""
import argparse
import json
import math
import sys
import time
import random
import torch
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import numpy as np
import pybullet as p

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from simulation.paint_environment import PaintEnvironment, PaintConfig
from simulation.room_config import RoomSampler
from simulation.robot import Robot
from simulation.paint_can import PaintCan
from mapping.coord_wall_map import CoordWallMap
from planning.motion_planner import MotionPlanner, MotionPlanningError
from evaluation.coverage import coverage_percentage
from evaluation.path_length import PathLengthTracker
from train_td3_wall import target_mask, solve_joints, center_bounds, PPM, DT, INSET, stage3_sampled_ik_mask
from train_stage3_goal_dagger_v4 import Stage3TrainEnv
from stage3_bottom_expert import Stage3FullWallSweep
from stable_baselines3 import SAC
from planning.motion_planner import CollisionQueryError


def _timed_motion_planner_method(method):
    @wraps(method)
    def timed(self, *args, **kwargs):
        top_level = self._planning_depth == 0
        if top_level:
            started = time.perf_counter()
            method_name = method.__name__
        self._planning_depth += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._planning_depth -= 1
            if top_level:
                elapsed = time.perf_counter() - started
                self._planning_calls += 1
                self._planning_seconds += elapsed
                stats = self._planning_by_method.setdefault(
                    method_name, {'calls': 0, 'total_seconds': 0.})
                stats['calls'] += 1
                stats['total_seconds'] += elapsed
    return timed


class DaggerMotionPlanner(MotionPlanner):
    """Keep shared planning rules while retrying transient PyBullet link queries."""
    def __init__(self, robot, collision_bodies, physics_client_id):
        super().__init__(robot, collision_bodies)
        self.physics_client_id = int(physics_client_id)
        self._planning_depth = 0
        self._planning_calls = 0
        self._planning_seconds = 0.
        self._planning_by_method = {}
        self._reported_collision_retry = False
        self._reported_collision = False
        self.collision_reporting_enabled = False
        self.paint_can_body_ids = set()
        self._reported_contact_tolerance = False
        self.wall_recovery_body = None

    def enable_collision_reporting(self):
        self.collision_reporting_enabled = True

    def register_paint_can(self, paint_can):
        super().register_paint_can(paint_can)
        self.paint_can_body_ids.update(map(int,paint_can.collision_body_ids))

    def _collision_state(self, allowed_contact_body, executing):
        details = {
            'client': self.physics_client_id,
            'robot_body': int(self.robot.id),
            'allowed_contact_body': (None if allowed_contact_body is None
                                     else int(allowed_contact_body)),
            'roller_link': int(self.robot.roller_link_index),
            'tolerated_contacts': [],
            'violations': [],
        }
        a, b = self.robot.get_roller_axis_segment()
        axis_z = max(-1., min(1., (a[2]-b[2])/self.robot.ROLLER_LENGTH))
        center_z = (a[2]+b[2])/2
        bottom = (center_z - self.robot.ROLLER_LENGTH/2*abs(axis_z)
                  - self.robot.ROLLER_RADIUS*math.sqrt(max(0.,1-axis_z**2)))
        clearance = .003 if executing else self.FLOOR_CLEARANCE
        details['roller_floor_clearance_mm'] = float(
            (bottom-self.floor_z-clearance)*1000)
        details['floor_violation'] = bottom < self.floor_z+clearance-1e-6

        for body in self.collision_bodies:
            for point in self._collision_points(body):
                link, distance = int(point[3]), float(point[8])
                if link == self.robot.roller_link_index:
                    standard_tolerance = (self.WALL_PENETRATION
                                          if body == allowed_contact_body else .0005)
                else:
                    standard_tolerance = .002
                if distance >= -standard_tolerance:
                    continue
                link_name = ('base' if link == -1 else p.getJointInfo(
                    self.robot.id,link,
                    physicsClientId=self.physics_client_id)[12].decode('utf-8'))
                tolerance = standard_tolerance
                kind = 'body_contact'
                if body == allowed_contact_body and link == self.robot.roller_link_index:
                    tolerance = .010 if self.wall_recovery_body == body else .005
                    kind = ('wall_contact_recovery' if self.wall_recovery_body == body
                            else 'active_roller_wall')
                elif (body in self.paint_can_body_ids and link_name == 'base_yaw_link'):
                    tolerance = .003 if executing else .002
                    kind = 'can_chassis'
                contact = {
                    'kind':kind,
                    'body_id':int(body),
                    'body_name':p.getBodyInfo(
                        body,physicsClientId=self.physics_client_id)[1].decode('utf-8'),
                    'robot_link':link,
                    'robot_link_name':link_name,
                    'distance_mm':distance*1000,
                    'allowed_penetration_mm':tolerance*1000,
                }
                if distance < -tolerance:
                    details['violations'].append(contact)
                else:
                    details['tolerated_contacts'].append(contact)
        return details

    def _emit_collision_detail(self, details, context=None):
        if context is not None:
            details['context'] = context
        print('[collision-detail] '+json.dumps(details),flush=True)

    def report_predicted_collision(self, joints, allowed_contact_body, context):
        saved = self._save_joints()
        try:
            self._apply_joints(joints)
            details = self._collision_state(allowed_contact_body,executing=False)
            self._emit_collision_detail(details,context)
        finally:
            self._restore_joints(saved)

    @_timed_motion_planner_method
    def plan_motion(self, target_position, orientation_mode="vertical",
                    allowed_contact_body=None):
        return super().plan_motion(
            target_position, orientation_mode=orientation_mode,
            allowed_contact_body=allowed_contact_body)

    @_timed_motion_planner_method
    def plan_transit_motion(self, target_position):
        return super().plan_transit_motion(target_position)

    @_timed_motion_planner_method
    def plan_retraction_motion(self, target_position, allowed_contact_body=None):
        return super().plan_retraction_motion(
            target_position, allowed_contact_body=allowed_contact_body)

    @_timed_motion_planner_method
    def plan_to_joint_configuration(self, target_joints):
        return super().plan_to_joint_configuration(target_joints)

    def planning_timing(self):
        methods = {
            name: {
                'calls': stats['calls'],
                'total_seconds': stats['total_seconds'],
                'mean_seconds': stats['total_seconds'] / stats['calls'],
            }
            for name, stats in self._planning_by_method.items()
        }
        return {
            'calls': self._planning_calls,
            'total_seconds': self._planning_seconds,
            'mean_seconds': (
                self._planning_seconds / self._planning_calls
                if self._planning_calls else 0.),
            'top_level_methods': methods,
        }

    @_timed_motion_planner_method
    def plan_joint_route(self, target_joints, allowed_contact_body=None,
                         max_iterations=1200):
        """Find a collision-checked route to an exact wall-frame IK solution."""
        start = list(map(float,self.robot.get_joint_states()))
        goal = list(map(float,target_joints))
        if (len(goal) != len(self.robot.joint_indices) or
                not all(math.isfinite(q) and lo <= q <= hi for q,lo,hi in
                        zip(goal,self.robot.lower_limits,self.robot.upper_limits))):
            raise MotionPlanningError('Invalid saved painting joint configuration')
        if self._predicted_collision(start,allowed_contact_body):
            raise MotionPlanningError('Refill return starts in collision')
        if self._predicted_collision(goal,allowed_contact_body):
            raise MotionPlanningError('Saved painting pose is collision-obstructed')

        def edge_clear(a,b):
            return all(not self._predicted_collision(q,allowed_contact_body)
                       for q in self._interpolate(a,b))

        direct = self._interpolate(start,goal)
        if all(not self._predicted_collision(q,allowed_contact_body) for q in direct):
            return direct

        scales = (.5,.5,1.,.7,.35,1.,.7)
        def distance(a,b):
            return max(abs(x-y)/scale for x,y,scale in zip(a,b,scales))

        forward = [(start,None)]
        backward = [(goal,None)]

        def extend(tree,target):
            nearest = min(range(len(tree)),key=lambda i:distance(tree[i][0],target))
            source = tree[nearest][0]
            gap = distance(source,target)
            if gap < 1e-9:
                return nearest,True
            fraction = min(1.,.5/gap)
            candidate = [a+fraction*(b-a) for a,b in zip(source,target)]
            if not edge_clear(source,candidate):
                return None,False
            tree.append((candidate,nearest))
            return len(tree)-1,fraction == 1.

        def root_path(tree,index):
            path = []
            while index is not None:
                joints,parent = tree[index]
                path.append(joints)
                index = parent
            return path[::-1]

        rng = random.Random(5478)
        lower,upper = self.robot.lower_limits,self.robot.upper_limits
        for iteration in range(max_iterations):
            tree_a,tree_b = ((forward,backward) if iteration % 2 == 0
                             else (backward,forward))
            sample = (list(rng.choice(tree_b)[0]) if rng.random() < .2 else
                      [rng.uniform(lo,hi) for lo,hi in zip(lower,upper)])
            index_a,_ = extend(tree_a,sample)
            if index_a is None:
                continue
            meeting = tree_a[index_a][0]
            for _ in range(64):
                index_b,connected = extend(tree_b,meeting)
                if index_b is None:
                    break
                if not connected:
                    continue
                if tree_a is forward:
                    path = (root_path(forward,index_a)
                            + root_path(backward,index_b)[-2::-1])
                else:
                    path = (root_path(backward,index_a)
                            + root_path(forward,index_b)[-2::-1])
                for _ in range(60):
                    if len(path) <= 2:
                        break
                    i,j = sorted(rng.sample(range(len(path)),2))
                    if j > i+1 and edge_clear(path[i],path[j]):
                        path[i+1:j] = []
                trajectory = []
                for first,second in zip(path,path[1:]):
                    trajectory.extend(self._interpolate(first,second))
                if not trajectory:
                    trajectory = [start]
                if (any(self._predicted_collision(q,allowed_contact_body)
                        for q in trajectory) or
                        not all(math.isclose(a,b,abs_tol=1e-3)
                                for a,b in zip(trajectory[-1],goal))):
                    raise MotionPlanningError(
                        'Refill return search failed final path validation')
                return trajectory
        raise MotionPlanningError(
            f'No collision-free refill return route within {max_iterations} iterations')

    def current_collision(self, allowed_contact_body=None, executing=False):
        try:
            details = self._collision_state(allowed_contact_body,executing)
        except p.error as first_error:
            details = {
                'client': self.physics_client_id,
                'connected': bool(p.isConnected(self.physics_client_id)),
                'body_id': int(self.robot.id),
                'roller_link': int(self.robot.roller_link_index),
            }
            if not details['connected']:
                raise RuntimeError(
                    f'PyBullet disconnected during collision checking: {details}'
                ) from first_error
            try:
                p.getBodyInfo(self.robot.id, physicsClientId=self.physics_client_id)
                joint_count = int(p.getNumJoints(
                    self.robot.id, physicsClientId=self.physics_client_id))
                details['joint_count'] = joint_count
                if not 0 <= self.robot.roller_link_index < joint_count:
                    details['link_index_valid'] = False
                    raise RuntimeError(
                        f'Invalid roller link during collision checking: {details}'
                    ) from first_error
                details['roller_link_name'] = p.getJointInfo(
                    self.robot.id, self.robot.roller_link_index,
                    physicsClientId=self.physics_client_id)[12].decode('utf-8')
                p.performCollisionDetection(physicsClientId=self.physics_client_id)
                details = self._collision_state(allowed_contact_body,executing)
            except p.error as retry_error:
                details['retry_error'] = str(retry_error)
                raise RuntimeError(
                    f'PyBullet collision query failed after refresh: {details}'
                ) from retry_error
            if not self._reported_collision_retry:
                print('[collision-query-retried] '+json.dumps(details),flush=True)
                self._reported_collision_retry = True
        colliding = details['floor_violation'] or bool(details['violations'])
        if details['tolerated_contacts'] and not self._reported_contact_tolerance:
            print('[collision-tolerance] '+json.dumps({
                'reason':'small numerical contact at an explicitly tolerated pair',
                'contacts':details['tolerated_contacts'],
            }),flush=True)
            self._reported_contact_tolerance = True
        if (colliding and self.collision_reporting_enabled and executing
                and not self._reported_collision):
            self._reported_collision = True
            self._emit_collision_detail(details)
        return colliding


class TimedStage3FullWallSweep(Stage3FullWallSweep):
    def action(self, env):
        started = time.perf_counter()
        try:
            return super().action(env)
        finally:
            env.goal_planner_calls += 1
            env.goal_planner_seconds += time.perf_counter() - started


class WorldFrameStage3(Stage3TrainEnv):
    """Stage3 observation/action interface backed by one world-frame robot.

    Stage3 uses wall-local prismatic coordinates; transform the seven joint
    targets and state features, never the physical robot base/URDF.
    """
    def __init__(self, walls, episode_steps=30000):
        super().__init__(walls, episode_steps=episode_steps)
        self.goal_planner_calls = 0
        self.goal_planner_seconds = 0.

    def attach(self, scene, robot, planner, wall, max_steps, openings):
        self.sim, self.robot, self.planner, self.wall = scene, robot, planner, wall
        scene.stage3_env = self
        self._client = scene.client
        self.depth = scene.room.depth if wall.name in ('south', 'north') else scene.room.width
        self.config.max_episode_steps = max_steps
        self.target = target_mask(wall.canvas, self.depth)
        self.target_pixels = int(self.target.sum())
        if not self.target_pixels:
            raise RuntimeError('Empty target mask')
        self.target_area = self.target_pixels / PPM**2
        self.previous_action = np.zeros(4, np.float32)
        self.steps = self.last_progress_step = self.idle_steps = 0
        self.area = self.meaningful_area = self.episode_return = 0.
        self.contact = self.done = False
        self.reason = 'running'
        self.reward_metrics = {}
        self.hud_ids = []
        self.command_status = 'reset'
        self.servo_scale = 1.
        self.rejected_steps = self.recovery_steps = 0
        self.joint_target_error = np.zeros(7)
        self.goal_uv = np.zeros(2)
        self._goal_features = np.zeros(16, np.float32)
        self._cached_action = np.zeros(4, np.float32)
        self.expert = None
        self.openings = openings
        self._mask_reference = None
        self.step_paint_distance = 0.
        self._distance_previous = None
        self._in_drive = False
        self._last_feasible_ik_command = None
        self._last_feasible_ik_joints = None
        self._wall_contact_guard = False
        self._floor_clearance_guard = False
        self._validate_robot()

    def finish_attach(self):
        # CurriculumEnv.reset refines the target after the physical starting
        # pose is established. Keep that ordering and the original planner.
        original_pixels = self.target_pixels
        self._mask_reference = np.asarray(self.robot.get_joint_states(),float)
        try:
            self.target, stats = stage3_sampled_ik_mask(self)
        finally:
            self._mask_reference = None
        self.target_pixels = int(self.target.sum())
        if not self.target_pixels:
            raise RuntimeError('No reachable target pixels')
        self.target_area = self.target_pixels / PPM**2
        self.stage3_mask_diagnostics = dict(original_target_pixels=original_pixels,
                                          candidate_target_pixels=self.target_pixels,
                                          ik_sampling=stats)
        self._update_goal()
        self.expert = TimedStage3FullWallSweep(self)
        self._prepare()
        self._last_feasible_ik_command = np.asarray(self.command, float).copy()
        self._last_feasible_ik_joints = np.asarray(self.robot.get_joint_states(), float).copy()
        return self._obs()

    def _record_path_position(self, position):
        tracker = getattr(self.sim, 'path_tracker', None)
        if tracker is None:
            return
        position = np.asarray(position, float)
        previous = getattr(self.sim, '_last_path_position', None)
        if previous is not None:
            tracker.update(previous, position)
        self.sim._last_path_position = position.copy()

    def _pose(self):
        try:
            pose = super()._pose()
        except RuntimeError as exc:
            if not str(exc).startswith('Roller pose query failed:'):
                raise
            client = self.sim.client
            details = {'client':client,'connected':bool(p.isConnected(client)),
                       'body_id':int(self.robot.id),
                       'roller_link':int(self.robot.roller_link_index)}
            try:
                details['body_count'] = int(p.getNumBodies(physicsClientId=client))
                details['joint_count'] = int(p.getNumJoints(self.robot.id,physicsClientId=client))
                if 0 <= self.robot.roller_link_index < details['joint_count']:
                    details['roller_link_name'] = p.getJointInfo(
                        self.robot.id,self.robot.roller_link_index,
                        physicsClientId=client)[12].decode('utf-8')
                    p.performCollisionDetection(physicsClientId=client)
                    try:
                        pose = super()._pose()
                    except RuntimeError as retry_error:
                        details['retry_error'] = str(retry_error)
                    else:
                        self._record_path_position(pose[0])
                        return pose
                else:
                    details['link_index_valid'] = False
            except p.error as diagnostic_error:
                details['diagnostic_error'] = str(diagnostic_error)
            raise RuntimeError(
                f'{exc}; PyBullet state={json.dumps(details)}') from exc
        self._record_path_position(pose[0])
        return pose

    def step(self, action):
        try:
            return super().step(action)
        finally:
            planner = getattr(self,'planner',None)
            if isinstance(planner,DaggerMotionPlanner):
                planner.wall_recovery_body = None

    def local_joints(self, q=None):
        q = np.array(self.robot.get_joint_states() if q is None else q, float)
        world = np.array([q[0], q[1], 0.]) - self.wall.origin
        q[0] = world @ self.wall.u_dir
        q[1] = world @ self.wall.normal
        q[2] = (q[2] - self.wall.yaw + math.pi) % (2*math.pi)-math.pi
        return q

    def world_joints(self, q, reference=None):
        q = np.array(q, float)
        xy = self.wall.origin + q[0]*self.wall.u_dir + q[1]*self.wall.normal
        q[0:2] = xy[:2]
        yaw = (q[2] + self.wall.yaw + math.pi) % (2*math.pi)-math.pi
        lo, hi = self.robot.lower_limits[2], self.robot.upper_limits[2]
        choices = [float(np.clip(a,lo,hi)) for a in (yaw,yaw+2*math.pi,yaw-2*math.pi)
                   if lo-1e-5 <= a <= hi+1e-5]
        reference_yaw = self.robot.get_joint_states()[2] if reference is None else reference[2]
        q[2] = min(choices,key=lambda a:abs(a-reference_yaw)) if choices else yaw
        return q

    def _ik(self, command, previous=None):
        command = np.asarray(command, float)
        result = None
        u0,u1,v0,v1 = center_bounds(self.wall.length,self.wall.height,command[3])
        if u0<=command[0]<=u1 and v0<=command[1]<=v1 and .0595<=command[2]<=.16:
            if previous is None:
                previous = getattr(self,'_mask_reference',None)
                if previous is None:
                    previous = self.robot.get_joint_states()
            prev_local = self.local_joints(previous)
            local = solve_joints(command,self.wall.length,self.depth,prev_local)
            if local is not None:
                world = self.world_joints(local, previous)
                lo,hi=np.asarray(self.robot.lower_limits),np.asarray(self.robot.upper_limits)
                if np.all(world>=lo-1e-6) and np.all(world<=hi+1e-6):
                    result = world
        if result is not None:
            if self._in_drive:
                self._last_feasible_ik_command = command.copy()
                self._last_feasible_ik_joints = result.copy()
            return result
        if (self._in_drive and self._last_feasible_ik_command is not None and
                np.allclose(command,self._last_feasible_ik_command,rtol=0.,atol=1e-10)):
            return self._last_feasible_ik_joints.copy()
        return None

    def _drive(self, action):
        action = np.asarray(action, np.float32)
        pose, axis, endpoint_a, endpoint_b = self._pose()
        center_z = (endpoint_a[2] + endpoint_b[2]) / 2
        axis_z = max(-1., min(1., (endpoint_a[2] - endpoint_b[2]) / self.robot.ROLLER_LENGTH))
        roller_bottom = (center_z - self.robot.ROLLER_LENGTH / 2 * abs(axis_z)
                         - self.robot.ROLLER_RADIUS * math.sqrt(max(0., 1 - axis_z**2)))
        floor_z = float(self.planner.floor_z)
        if not self._floor_clearance_guard and roller_bottom < floor_z + .012:
            self._floor_clearance_guard = True
        if self._floor_clearance_guard:
            if roller_bottom >= floor_z + .013:
                self._floor_clearance_guard = False
            else:
                _, _, v0, v1 = center_bounds(
                    self.wall.length, self.wall.height, float(self.command[3]))
                required_center_height = (
                    floor_z + .004
                    + self.robot.ROLLER_LENGTH / 2 * abs(axis_z)
                    + self.robot.ROLLER_RADIUS * math.sqrt(max(0., 1 - axis_z**2))
                )
                self.command = np.asarray(self.command, float).copy()
                self.command[1] = float(np.clip(
                    max(self.command[1], required_center_height), v0, v1))
                action = action.copy()
                action[1] = max(0., float(action[1]))
                candidate_v = self.command[1] + float(action[1]) * .10 * DT
                candidate_theta = (float(self.command[3]) + float(action[3]) * .70 * DT
                                   + math.pi) % (2 * math.pi) - math.pi
                vertical_axis = abs(math.sin(candidate_theta))
                required_center_height = (
                    floor_z + .004
                    + self.robot.ROLLER_LENGTH / 2 * vertical_axis
                    + self.robot.ROLLER_RADIUS * math.sqrt(max(0., 1 - vertical_axis**2))
                )
                if candidate_v < required_center_height + .001:
                    action[3] = 0.
        endpoint_clearance = min(
            float(self.wall.signed_distance(point)) - self.robot.ROLLER_RADIUS
            for point in (endpoint_a, endpoint_b))
        unsafe_contact = (endpoint_clearance < -0.0015 or
                          abs(float(axis[2])) > .025)
        if unsafe_contact and not self._wall_contact_guard:
            self._wall_contact_guard = True
            self.command = np.asarray(self.command, float).copy()
            self.command[:2] = pose[:2]
            self.command[2] = float(np.clip(max(self.command[2], pose[2] + .015),
                                            .0595, .16))
            self.previous_action = np.zeros(4, np.float32)
        wall_recovery_active = self._wall_contact_guard
        if self._wall_contact_guard:
            if not unsafe_contact and endpoint_clearance >= .006 and abs(float(axis[2])) <= .015:
                self._wall_contact_guard = False
            action = np.zeros(4, np.float32)
        self._in_drive = True
        try:
            try:
                rejected = super()._drive(action)
            except RuntimeError as exc:
                if str(exc) != 'Last feasible controller command has no IK solution':
                    raise
                qnow = np.asarray(self.robot.get_joint_states(), float)
                theta = math.atan2(float(axis[1]), float(axis[0]))
                u0,u1,v0,v1 = center_bounds(self.wall.length,self.wall.height,theta)
                command = np.array([
                    np.clip(pose[0],u0,u1),
                    np.clip(pose[1],v0,v1),
                    np.clip(max(float(pose[2]),.0595),.0595,.16),
                    theta,
                ], float)
                self.command = command
                self._last_feasible_ik_command = command.copy()
                self._last_feasible_ik_joints = qnow.copy()
                self.previous_action = np.zeros(4, np.float32)
                self.joint_target_error = np.zeros(7)
                self.servo_scale = 0.
                self.rejected_steps += 1
                self.recovery_steps += 1
                self._wall_contact_guard = False
                self.robot.set_joint_targets(qnow)
                self.command_status = 'ik_recovery'
                rejected = True
        finally:
            self._in_drive = False
        self.planner.wall_recovery_body = (
            self.wall.body_id if wall_recovery_active else None)
        if self._floor_clearance_guard:
            self.command_status = 'floor_clearance_recovery'
        elif self._wall_contact_guard:
            self.command_status = 'wall_contact_recovery'
        return rejected

    def _obs(self):
        # Base Stage3 observation is canonical wall-local except the joint state.
        obs = super()._obs()
        states = p.getJointStates(self.robot.id,self.robot.joint_indices)
        q = self.local_joints([s[0] for s in states])
        dq = np.array([s[1] for s in states],float)
        # Local translation velocities are rotated; yaw velocity is invariant.
        dq[0],dq[1] = float(dq[0]*self.wall.u_dir[0]+dq[1]*self.wall.u_dir[1]), float(dq[0]*self.wall.normal[0]+dq[1]*self.wall.normal[1])
        lo,hi=np.array(self.robot.lower_limits,float),np.array(self.robot.upper_limits,float)
        lo[:2]=[INSET,INSET];hi[:2]=[self.wall.length-INSET,self.depth-INSET]
        obs['state'][:7]=(2*(q-lo)/(hi-lo)-1).astype(np.float32)
        obs['state'][7:14]=dq.astype(np.float32)
        return obs

    # _drive, _prepare, _pose, _contact_state, _info and the goal planner
    # are inherited unchanged. _ik and the first 14 observation entries only
    # adapt the fixed world-frame robot to the original wall-local interface.

    def _paint(self, draw):
        painted = super()._paint(draw)
        roller = np.asarray(self.robot.get_roller_position(),float)
        if painted and self._distance_previous is not None:
            self.step_paint_distance += float(np.linalg.norm(roller-self._distance_previous))
        self._distance_previous = roller
        return painted

    def step(self, action):
        # Call the real Stage3TrainEnv -> GoalEnv -> CurriculumEnv ->
        # WholeWallEnv chain. No replacement rewards, clocks or phase changes.
        self.step_paint_distance = 0.
        self._distance_previous = np.asarray(self.robot.get_roller_position(),float)
        return super().step(action)

    def _draw_reward_hud(self, info, reward):
        # Display-only optimisation: the model and physics do not read the HUD.
        if not getattr(self,'fast_gui',False):
            return super()._draw_reward_hud(info,reward)


def paint_diagnostics(env, action, goal_action):
    pose,axis,_,_ = env._pose()
    wanted_axis = np.array([math.cos(env.command[3]),math.sin(env.command[3]),0.])
    contact = env._contact_state()
    contact['paint_contact'] = bool(contact['paint_contact'])
    return {'pose_uvn':pose.tolist(),'command':env.command.tolist(),
            'position_error_m':float(np.linalg.norm(pose-env.command[:3])),
            'axis_alignment':abs(float(axis @ wanted_axis)),
            'actor_action':np.asarray(action).tolist(),
            'goal_action_before_step':np.asarray(goal_action).tolist(),
            'controller':env.command_status,'servo_scale':float(env.servo_scale),
            'joint_target_error':np.asarray(env.joint_target_error).tolist(),
            'contact':contact,
            'no_progress_steps':int(env.steps-env.last_progress_step),
            'remaining_steps':int(env.expert._stall_budget(env)[0]),
            'lane':int(env.expert.index),'lanes':len(env.expert.lanes),
            'phase':env.expert.phase,'cleanup':bool(env.expert.cleanup_mode),
            'cleanup_phase':env.expert.cleanup_phase,
            'boundary_skips':int(getattr(env.expert,'boundary_skips',0)),
            'wall_contact_guard':bool(env._wall_contact_guard),
            'floor_clearance_guard':bool(env._floor_clearance_guard)}


class FastScene(PaintEnvironment):
    """Keep every physics tick; limit expensive GUI work by wall-clock time."""
    def set_display_rate(self, hz):
        self.display_hz = hz
        self.next_display = 0.

    def step(self):
        if not getattr(self, 'display_hz', 0):
            return super().step()
        # The supplied PaintEnvironment.step only advances physics, refreshes
        # textures, and handles camera keys. Physics is never skipped here.
        p.stepSimulation(physicsClientId=self.client)
        self._steps += 1
        now = time.perf_counter()
        if self.gui and now >= self.next_display:
            self.flush_textures()
            self._camera_keys()
            self.next_display = now + 1./self.display_hz


def lift_all(scene):
    for wall in scene.walls:
        wall.canvas.lift()


def checked_collision(robot, planner, scene, allowed=None):
    try:
        return planner.current_collision(allowed, executing=True)
    except p.error as exc:
        details = {'client':int(scene.client),'connected':bool(p.isConnected(scene.client)),
                   'body_id':int(robot.id),'roller_link':int(robot.roller_link_index)}
        if not details['connected']:
            raise RuntimeError(f'PyBullet disconnected during planned travel: {details}') from exc
        try:
            p.getBodyInfo(robot.id, physicsClientId=scene.client)
            joint_count = int(p.getNumJoints(robot.id, physicsClientId=scene.client))
            details['joint_count'] = joint_count
            if not 0 <= robot.roller_link_index < joint_count:
                details['link_index_valid'] = False
                raise RuntimeError(f'Invalid roller link during planned travel: {details}') from exc
            details['roller_link_name'] = p.getJointInfo(
                robot.id, robot.roller_link_index, physicsClientId=scene.client
            )[12].decode('utf-8')
            p.performCollisionDetection(physicsClientId=scene.client)
            result = planner.current_collision(allowed, executing=True)
            print('[collision-query-retried] '+json.dumps(details),flush=True)
            return result
        except p.error as retry_error:
            details['retry_error'] = str(retry_error)
            raise RuntimeError(
                f'PyBullet collision query failed after refresh: {details}') from retry_error


def track_current_tool_position(scene):
    if getattr(scene, 'path_tracker', None) is not None:
        env = getattr(scene, 'stage3_env', None)
        if env is not None:
            env._pose()


def print_room_coverage_summary(scene, wall_maps, wall_order, path_tracker, elapsed):
    total_cells = sum(wall_maps[name].total_cells() for name in wall_order)
    total_painted = sum(wall_maps[name].painted_count() for name in wall_order)
    overall_coverage = 100. * total_painted / total_cells if total_cells else 0.
    per_wall = {}
    for name in wall_order:
        wall_map = wall_maps[name]
        coverage = coverage_percentage(wall_map)
        cells_below_80, below_80_total = wall_map.uncovered_cells(limit=20, threshold=.8)
        per_wall[name] = {
            'coverage_percent': coverage,
            'cells_below_80_total': below_80_total,
            'cells_below_80': [
                [int(row), int(col), float(fraction)]
                for row, col, fraction in cells_below_80
            ],
        }

    all_complete = len(per_wall) == len(wall_order) and all(
        per_wall[name]['coverage_percent'] >= 100. for name in wall_order)
    print("\n=== Room painting summary ===")
    print("All walls complete!" if all_complete else "Room painting stopped before full coverage.")
    print(f"Overall coverage: {overall_coverage:.1f}% ({total_painted}/{total_cells} cells)")
    for name in wall_order:
        stats = per_wall[name]
        if stats['coverage_percent'] < 100.:
            print(f"[{name}] incomplete wall: coverage={stats['coverage_percent']:.1f}%")
            uncovered = [
                (row, col, f'{fraction:.2%}')
                for row, col, fraction in stats['cells_below_80']
            ]
            if uncovered:
                print(f"[{name}] cells below 80% coverage: {uncovered}")
    room_wall_area, _ = scene.painted_area()
    per_wall_area = {name: float(room_wall_area[name]) for name in wall_order}
    total_area = sum(per_wall_area.values())
    print(f"Ground-truth painted area (from canvas): {total_area:.3f} m^2 {per_wall_area}")
    path_length = float(path_tracker.get_total_length())
    print(f"Total path length travelled: {path_length:.2f} m")
    print(f"Total runtime (environment init to painting finish): {elapsed:.2f} s")
    return {
        'overall_coverage_percent': overall_coverage,
        'painted_cells': total_painted,
        'total_cells': total_cells,
        'per_wall': per_wall,
        'ground_truth_painted_area_m2': total_area,
        'wall_painted_area_m2': per_wall_area,
        'total_path_length_m': path_length,
        'elapsed_seconds': elapsed,
    }


def run_path(robot, planner, scene, path, allowed=None, tolerance=None,
             final_check=None, diagnostics=None, label="planned travel"):
    """Execute checked waypoints without painting or changing policy state."""
    # X, Y and extension are metres; yaw, shoulder and wrists are radians.
    # A single 0.0005 threshold wrongly required 0.029-degree angular tracking.
    tolerances = np.broadcast_to(np.asarray(
        [.002,.002,.005,.005,.002,.005,.005] if tolerance is None else tolerance,
        dtype=float),(7,))
    if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
        raise ValueError('Joint tolerances must be finite and positive')
    path = [np.asarray(q, float) for q in path]
    if not path:
        raise MotionPlanningError('Planner returned an empty path')
    track_current_tool_position(scene)
    for index, waypoint in enumerate(path):
        # Check from the actual tracked pose, including reversed return routes.
        current = robot.get_joint_states()
        if any(planner._predicted_collision(q, allowed)
               for q in planner._interpolate(current, waypoint)):
            raise MotionPlanningError('Path from actual motor pose is obstructed')
        robot.set_joint_targets(waypoint)  # held by Bullet; no per-tick reissue
        stable_ticks = 0
        final_waypoint = index == len(path)-1
        intermediate_tolerances = np.array([.008,.008,.011,.011,.008,.025,.025])
        tol = tolerances if final_waypoint else np.maximum(tolerances,intermediate_tolerances)
        for _ in range(480):
            scene.step()
            if checked_collision(robot, planner, scene, allowed):
                raise MotionPlanningError('Collision during planned travel')
            error = np.asarray(robot.get_joint_states())-waypoint
            if final_waypoint and final_check is not None:
                # Contact forces can keep joints away from a penetrating IK
                # target. Require a stable, measured tool pose instead.
                reached = bool(np.max(np.abs(error)) <= .02 and final_check())
                stable_ticks = stable_ticks+1 if reached else 0
                if stable_ticks >= 8:
                    break
            elif np.all(np.abs(error) <= tol):
                break
        else:
            detail = {'stage':label,'waypoint':index+1,'waypoints':len(path),
                      'final_waypoint':final_waypoint,'joint_tolerance':tol.tolist(),
                      'max_joint_error':float(np.max(np.abs(error))),
                      'joint_error':error.tolist(),
                      'actual_joints':list(map(float,robot.get_joint_states())),
                      'target_joints':waypoint.tolist(),
                      'roller_position':list(map(float,robot.get_roller_position()))}
            if diagnostics is not None:
                detail['contact'] = diagnostics()
            raise MotionPlanningError('Motor tracking timeout: '+json.dumps(detail))
        track_current_tool_position(scene)


def retreat(env, robot, planner, scene, wall):
    lift_all(scene)
    start = np.asarray(robot.get_joint_states(), float)
    pose, axis, _, _ = env._pose()
    theta = math.atan2(float(axis[1]),float(axis[0]))
    candidates = [
        ('command', np.asarray(env.command, float).copy()),
        ('measured', np.array([pose[0],pose[1],.13,theta],float)),
        ('measured', np.array([pose[0],pose[1],.16,theta],float)),
    ]
    if planner.paint_cans:
        can_u,_ = wall.world_to_uv(planner.paint_cans[0].position)
        direction = -1. if can_u > pose[0] else 1.
        u0,u1,_,_ = center_bounds(wall.length,wall.height,theta)
        for offset in (.15,.30,.45,.60):
            command = np.array([
                np.clip(pose[0]+direction*offset,u0,u1),pose[1],.13,theta],float)
            candidates.append(('can-clearance',command))
    failures = []
    path = None
    for source, command in candidates:
        command[2] = .13 if source == 'command' else command[2]
        target = env._ik(command, previous=start)
        if target is None:
            failures.append(f'{source}: IK unavailable')
            continue
        candidate_path = planner._interpolate(start, target)
        collision_index = next((
            index for index, joints in enumerate(candidate_path)
            if planner._predicted_collision(joints, wall.body_id)
        ),None)
        if collision_index is not None:
            if isinstance(planner,DaggerMotionPlanner):
                planner.report_predicted_collision(
                    candidate_path[collision_index],wall.body_id,
                    f'retraction_{source}_normal_{command[2]:.3f}m')
            failures.append(f'{source} at normal={command[2]:.3f}m: obstructed')
            continue
        path = candidate_path
        if source != 'command':
            print('[retraction-replanned] '+json.dumps({
                'wall':wall.name,'source':source,'normal_m':float(command[2])}),flush=True)
        break
    if path is None and planner.paint_cans:
        can_u,_ = wall.world_to_uv(planner.paint_cans[0].position)
        base_u,_ = wall.world_to_uv([start[0],start[1],0.])
        direction = -1. if can_u > base_u else 1.
        u0,u1,_,_ = center_bounds(wall.length,wall.height,theta)
        for offset in (.15,.30,.45,.60,.90,1.20):
                shifted_u = float(np.clip(pose[0]+direction*offset,u0,u1))
                side_command = np.array(
                    [shifted_u,pose[1],max(float(pose[2]),.0595),theta],float)
                side_target = env._ik(side_command, previous=start)
                if side_target is None:
                    continue
                side_path = planner._interpolate(start,side_target)
                collision_index = next((
                    index for index,joints in enumerate(side_path)
                    if planner._predicted_collision(joints,wall.body_id)
                ),None)
                if collision_index is not None:
                    if isinstance(planner,DaggerMotionPlanner):
                        planner.report_predicted_collision(
                            side_path[collision_index],wall.body_id,
                            f'can-clearance-tangent_{offset:.2f}m')
                    continue
                retract_command = side_command.copy()
                retract_command[2] = .13
                retract_target = env._ik(retract_command, previous=side_target)
                if retract_target is None:
                    continue
                retract_path = planner._interpolate(side_target,retract_target)
                collision_index = next((
                    index for index,joints in enumerate(retract_path)
                    if planner._predicted_collision(joints,wall.body_id)
                ),None)
                if collision_index is not None:
                    if isinstance(planner,DaggerMotionPlanner):
                        planner.report_predicted_collision(
                            retract_path[collision_index],wall.body_id,
                            f'can-clearance-retract_{offset:.2f}m')
                    continue
                path = side_path+retract_path
                print('[retraction-replanned] '+json.dumps({
                    'wall':wall.name,'source':'can-clearance-staged',
                    'base_away_from_can_m':offset}),flush=True)
                break
    if path is None:
        target_position = wall.uv_to_world(pose[0],pose[1]) + .13*wall.normal
        path = planner.plan_transit_motion(target_position)
        print('[retraction-rerouted] '+json.dumps({
            'wall':wall.name,'failed_direct_paths':failures}),flush=True)
    run_path(robot, planner, scene, path, allowed=wall.body_id)
    if checked_collision(robot, planner, scene):
        raise MotionPlanningError('Retraction did not clear the wall')
    return [start, *path]


def joint_approach(robot, planner, scene, target, allowed=None):
    # Retain the supplied planner's interpolation resolution and collision rules.
    path = planner._interpolate(robot.get_joint_states(), target)
    if any(planner._predicted_collision(q, allowed) for q in path):
        raise MotionPlanningError('Painting pose approach is obstructed')
    run_path(robot, planner, scene, path, allowed=allowed, label="joint positioning")


def contact_start(env, robot, planner, scene):
    """Physically reach a seeded start drawn from the native reset distribution.

    Native reset teleports only at episode setup; this runner instead plans
    and executes the approach so the shared room and robot remain intact.
    """
    wall = env.wall
    failures = []
    for _ in range(100):
        cmd = np.array([env.np_random.uniform(.25,wall.length-.25),
                        env.np_random.uniform(.30,min(1.8,wall.height-.2)),
                        .0595,env.np_random.choice([0.,math.pi/2])],float)
        if not env._solid_at(cmd[:2]):
            continue
        q = env._ik(cmd)
        clear_cmd = cmd.copy(); clear_cmd[2] = .16
        clear_q = env._ik(clear_cmd)
        if q is None or clear_q is None:
            continue
        # Reject unreachable/colliding candidates before expensive transit.
        if planner._predicted_collision(clear_q) or planner._predicted_collision(q,wall.body_id):
            continue
        point = wall.uv_to_world(cmd[0],max(.35,cmd[1])) + .16*wall.normal
        try:
            path = planner.plan_transit_motion(point)
            orient_path = planner._interpolate(path[-1], clear_q)
            contact_path = planner._interpolate(clear_q, q)
            if any(planner._predicted_collision(j) for j in orient_path):
                raise MotionPlanningError('Clear-pose orientation path is obstructed')
            if any(planner._predicted_collision(j,wall.body_id) for j in contact_path):
                raise MotionPlanningError('Final contact approach is obstructed')
        except CollisionQueryError:
            raise
        except MotionPlanningError as exc:
            failures.append(str(exc)); continue
        # Execution errors are fatal; never continue from an unverified pose.
        run_path(robot,planner,scene,path)
        run_path(robot,planner,scene,orient_path,label="pre-contact orientation")
        def contact_details():
            pose,axis,_,_ = env._pose()
            desired_axis = np.array([math.cos(cmd[3]),math.sin(cmd[3]),0.])
            return dict(env._contact_state(),
                        position_error_m=float(np.linalg.norm(pose-cmd[:3])),
                        axis_alignment=abs(float(axis @ desired_axis)))
        def contact_ready():
            detail = contact_details()
            return (detail['paint_contact'] and detail['position_error_m'] <= .002
                    and detail['axis_alignment'] >= .999)
        run_path(robot,planner,scene,contact_path,allowed=wall.body_id,
                 final_check=contact_ready,diagnostics=contact_details,
                 label='initial wall contact')
        env.command = cmd
        env.verify_pose(cmd)
        env.contact = env._contact_state()['paint_contact']
        if not env.contact:
            raise MotionPlanningError('Approach reached pose without verified painting contact')
        lift_all(scene)
        scene.focus_camera_on_wall(wall.name)
        print('[wall-ready] '+json.dumps({'wall':wall.name,'command':cmd.tolist()}),flush=True)
        return
    raise MotionPlanningError('No safe native-distribution start: '+repr(failures[-3:]))


def traditional_dip_and_resume(env, robot, planner, scene, can):
    """Use original planner and replan the return to the interrupted pose."""
    saved_q = np.asarray(robot.get_joint_states(),float)
    saved_position = np.asarray(robot.get_roller_position(),float)
    saved_axis = np.subtract(*robot.get_roller_axis_segment())/robot.ROLLER_LENGTH
    saved_pose,_,_,_ = env._pose()
    # env.command, expert, goals, prior action and paint timers stay untouched.
    print('[dip] saving painting pose and withdrawing',flush=True)
    retreat(env,robot,planner,scene,env.wall)
    approach = can.get_approach_position(robot.ROLLER_LENGTH,robot.ROLLER_RADIUS,.05)
    out_transit = planner.plan_transit_motion(approach)
    run_path(robot,planner,scene,out_transit)
    # Restore the can approach before planning a fresh return to the wall.
    approach_q = np.asarray(robot.get_joint_states(),float)
    run_path(robot,planner,scene,planner.plan_motion(can.get_dip_position(),orientation_mode='vertical'))
    can.dipped = False
    a,b = robot.get_roller_axis_segment()
    if not can.try_dip(a,b,robot.ROLLER_RADIUS):
        raise MotionPlanningError('Refill failed full-barrel containment check')
    run_path(robot,planner,scene,planner.plan_motion(approach,orientation_mode='vertical'))
    joint_approach(robot,planner,scene,approach_q)
    saved_u,saved_v = saved_pose[:2]
    clear_v = min(max(float(saved_v),.60),env.wall.height-.25)
    return_clear_position = (
        env.wall.uv_to_world(saved_u,clear_v) + .16*env.wall.normal)
    return_path = planner.plan_transit_motion(return_clear_position)
    run_path(robot,planner,scene,return_path,label='return transit to wall-clear pose')

    def resumed_pose_details():
        position = np.asarray(robot.get_roller_position(),float)
        axis = np.subtract(*robot.get_roller_axis_segment())/robot.ROLLER_LENGTH
        return {
            'position_error_m':float(np.linalg.norm(position-saved_position)),
            'axis_alignment':abs(float(axis @ saved_axis)),
            'roller_position':position.tolist(),
            'saved_position':saved_position.tolist(),
            'roller_axis':axis.tolist(),
            'saved_axis':saved_axis.tolist(),
        }
    def resumed_pose():
        details = resumed_pose_details()
        return (details['position_error_m'] <= .003
                and details['axis_alignment'] >= .999)
    resume_path = planner.plan_joint_route(
        saved_q,allowed_contact_body=env.wall.body_id)
    run_path(robot,planner,scene,resume_path,allowed=env.wall.body_id,
             final_check=resumed_pose,diagnostics=resumed_pose_details,
             label='return to saved painting pose')
    position_error = float(np.linalg.norm(np.asarray(robot.get_roller_position())-saved_position))
    axis = np.subtract(*robot.get_roller_axis_segment())/robot.ROLLER_LENGTH
    joint_error = float(np.max(np.abs(np.asarray(robot.get_joint_states())-saved_q)))
    if not resumed_pose():
        raise MotionPlanningError('Returned pose does not match interrupted painting pose')
    lift_all(scene)  # never draw a stripe between the can and wall
    env.contact = env._contact_state()['paint_contact']
    scene.focus_camera_on_wall(env.wall.name)
    print('[dip-resume] '+json.dumps({'wall':env.wall.name,'position_error_m':position_error,
                                    'max_joint_error':joint_error}),flush=True)


def place_can(robot, planner, scene):
    """Choose a free interior location before travel; never force travel via it."""
    w,d = scene.room.width,scene.room.depth
    for x,y in ((w*.5,d*.5),(w*.65,d*.5),(w*.35,d*.5),(w*.5,d*.7),(w*.5,d*.3)):
        can = PaintCan(position=(x,y,0.))
        planner.register_paint_can(can)
        if not planner.current_collision(executing=True):
            print('[can] '+json.dumps({
                'position':can.position,
                'collision_body_ids':list(can.collision_body_ids)}),flush=True)
            return can
        planner.paint_cans.remove(can)
        if isinstance(planner,DaggerMotionPlanner):
            planner.paint_can_body_ids.difference_update(
                map(int,can.collision_body_ids))
        for body in can.collision_body_ids:
            planner.collision_bodies.remove(body)
            p.removeBody(body)
    raise MotionPlanningError('No collision-free interior paint-can placement')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paint-actor',type=Path,required=True)
    ap.add_argument('--room-seed',type=int,default=None,
                    help='Seed a sampled room; omit to use RoomConfig.default()')
    ap.add_argument('--walls',nargs='+',choices=['south','east','north','west'],
                    default=['south','east','north','west'],help='Subset, always visited anticlockwise')
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--fast-gui',action='store_true',help='GUI with wall-clock-limited texture/camera updates')
    ap.add_argument('--display-hz',type=float,default=8.)
    ap.add_argument('--hold-gui',action='store_true')
    ap.add_argument('--max-actions',type=int,default=50000,help='Painting actions per wall')
    ap.add_argument('--target-coverage',type=float,default=.98,help='Fraction of reachable target, not entire wall')
    ap.add_argument('--log-every',type=int,default=500)
    ap.add_argument('--paint-distance-m',type=float,default=12.,help='Contact-travel budget between refills')
    ap.add_argument('--max-refills',type=int,default=30,help='Total room refill limit')
    ap.add_argument('--torch-threads',type=int,default=1)
    ap.add_argument('--out',type=Path,default=Path('sac_project/runs/dagger_continuous'))
    args = ap.parse_args()
    print('[runner] native-stage3-v5',flush=True)
    if not args.paint_actor.is_file(): ap.error('Missing checkpoint: '+str(args.paint_actor))
    if not 0 < args.target_coverage <= 1: ap.error('target-coverage must be in (0,1]')
    if any(x <= 0 for x in (args.max_actions,args.log_every,args.torch_threads)): ap.error('Counts must be positive')
    if not math.isfinite(args.paint_distance_m) or args.paint_distance_m <= 0: ap.error('Invalid paint distance')
    if not math.isfinite(args.display_hz) or args.display_hz <= 0: ap.error('Invalid display rate')
    if args.max_refills < 0: ap.error('max-refills must be nonnegative')
    args.gui |= args.fast_gui
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'diagnostics.jsonl').open('a') as log:
        log.write(json.dumps({'event':'run_start','runner':'native-stage3-v5',
                              'seed':args.room_seed,'unix_time':time.time()})+'\n')
    if args.room_seed is not None:
        random.seed(args.room_seed)
        np.random.seed(args.room_seed)
        torch.manual_seed(args.room_seed)
    torch.set_num_threads(args.torch_threads)
    actor = SAC.load(str(args.paint_actor),device='cpu')
    actor.policy.set_training_mode(False)
    if actor.observation_space['state'].shape != (56,) or actor.action_space.shape != (4,):
        ap.error('Expected the 56-state, 4-action Stage 3 checkpoint')
    scene = FastScene(gui=args.gui,paint=PaintConfig(patch_width=.004,px_per_m=PPM,
                      render_paint=args.gui),seed=args.room_seed,sampler=RoomSampler(l_shape_probability=0.))
    scene.set_display_rate(args.display_hz if args.fast_gui else 0)
    results=[]; env=None; planner=None; failure=None; total_refills=0; paint_distance=0.
    started=time.perf_counter()
    active_name=None
    order=[n for n in ('south','east','north','west') if n in args.walls]
    path_tracker=PathLengthTracker()
    scene.path_tracker=path_tracker
    scene._last_path_position=None
    cell_width,cell_height=2*Robot.ROLLER_RADIUS,Robot.ROLLER_LENGTH
    wall_maps={
        name:CoordWallMap(
            scene.walls_by_name[name],cell_width=cell_width,cell_height=cell_height,
            touch_spacing=2*Robot.ROLLER_RADIUS,line_spacing=Robot.ROLLER_LENGTH)
        for name in order
    }

    def save(coverage_report=None):
        per_wall,total = scene.painted_area()
        data={'room_seed':args.room_seed,'room':asdict(scene.room),'wall_order':order,
              'checkpoint':str(args.paint_actor),'results':results,'failure':failure,
              'active_wall':active_name,'total_refills':total_refills,
              'paint_distance_since_refill_m':paint_distance,
              'painted_area_m2':total,'wall_painted_area_m2':per_wall,
              'total_path_length_m':path_tracker.get_total_length(),
              'planner_timing':{
                  'stage3_goal_planner':{
                      'method':'Stage3FullWallSweep.action',
                      'calls':0 if env is None else env.goal_planner_calls,
                      'total_seconds':0. if env is None else env.goal_planner_seconds,
                      'mean_seconds':(
                          env.goal_planner_seconds/env.goal_planner_calls
                          if env is not None and env.goal_planner_calls else 0.),
                  },
                  'motion_planner':(
                      planner.planning_timing() if planner is not None else {
                          'calls':0,'total_seconds':0.,'mean_seconds':0.,
                          'top_level_methods':{}}),
              },
              'elapsed_seconds':time.perf_counter()-started,'persistent_room':True,
              'painting_core':'original_stage3','neural_painting':True,'dipping':'original_planner_reverse_route_resume',
              'completed':failure is None and len(results)==len(order) and all(r['reason']=='success' for r in results)}
        if coverage_report is not None:
            data['coverage_report']=coverage_report
        temporary=args.out/'summary.tmp'
        temporary.write_text(json.dumps(data,indent=2)+'\n')
        temporary.replace(args.out/'summary.json')
    try:
        w,d=scene.room.width,scene.room.depth
        robot=Robot(start_xy=(w/2,d/2),max_lift=scene.room.height,
                    x_range=(INSET,w-INSET),y_range=(INSET,d-INSET))
        planner=DaggerMotionPlanner(
            robot,[wall.body_id for wall in scene.walls]+[scene.floor_id],scene.client)
        planner.configure_environment(scene)
        if planner.current_collision(): raise MotionPlanningError('Initial robot collision')
        can=place_can(robot,planner,scene)
        planner.enable_collision_reporting()
        env=WorldFrameStage3(['south'],episode_steps=args.max_actions)
        env.gui=args.gui
        env.fast_gui=args.fast_gui
        # Never call env.reset(): it would destroy the shared world.
        print('[room] '+json.dumps({'seed':args.room_seed,'width':w,'depth':d,
              'height':scene.room.height,'anticlockwise':order}),flush=True)
        previous_wall=None
        for name in order:
            active_name=name
            if previous_wall is not None:
                retreat(env,robot,planner,scene,previous_wall)
            wall=scene.walls_by_name[name]
            wall_seed = None if args.room_seed is None else args.room_seed+order.index(name)
            env.np_random=np.random.default_rng(wall_seed)
            env.attach(scene,robot,planner,wall,args.max_actions,'stage3')
            env.config.success_fraction=args.target_coverage
            for key,space in env.observation_space.spaces.items():
                if key not in actor.observation_space.spaces or space.shape!=actor.observation_space[key].shape:
                    raise RuntimeError('Checkpoint observation mismatch: '+key)
            contact_start(env,robot,planner,scene)
            obs=env.finish_attach(); info={}; wall_refills=0
            print('[painting-core] '+json.dumps({
                'step':Stage3TrainEnv.step.__module__+'.'+Stage3TrainEnv.step.__qualname__,
                'drive':env._drive.__func__.__module__+'.'+env._drive.__func__.__qualname__,
                'goal_planner':Stage3FullWallSweep.__name__,
                'stall_seconds':env.config.stall_seconds,
                'idle_contact_seconds':env.config.idle_contact_seconds,
                'action_source':'Stage3 checkpoint only'}),flush=True)
            for step in range(args.max_actions):
                if paint_distance >= args.paint_distance_m:
                    if total_refills >= args.max_refills:
                        env.reason='refill_limit'; break
                    traditional_dip_and_resume(env,robot,planner,scene,can)
                    total_refills+=1; wall_refills+=1; paint_distance=0.
                    # Do not advance the planner again on the same control step.
                    obs=env._obs()
                goal_action = env._cached_action.copy()
                with torch.inference_mode():
                    action,_=actor.predict(obs,deterministic=True)
                action_source = 'stage3_checkpoint'
                obs,_,done,truncated,info=env.step(action)
                paint_distance+=env.step_paint_distance
                if (step+1)%args.log_every==0 or done or truncated:
                    diagnostic = paint_diagnostics(env,action,goal_action)
                    diagnostic['action_source'] = action_source
                    with (args.out/'diagnostics.jsonl').open('a') as log:
                        log.write(json.dumps(dict(wall=name,steps=step+1,**diagnostic))+'\n')
                    print('[control] '+json.dumps(diagnostic),flush=True)
                    print('[paint] '+json.dumps({'wall':name,'steps':step+1,
                          'target':info['target_coverage'],'whole':info['wall_coverage'],
                          'refills':total_refills,'paint_distance_m':paint_distance,
                          'elapsed_s':round(time.perf_counter()-started,1)}),flush=True)
                if done or truncated: break
            else:
                env.reason='action_limit'
            result={'wall':name,'steps':env.steps,'target_coverage':env.area/env.target_area,
                    'whole_wall_coverage':wall.canvas.painted_area/wall.canvas.paintable_area,
                    'reason':env.reason,'traditional_refills':wall_refills}
            results.append(result); save()
            print('[wall-result] '+json.dumps(result),flush=True)
            if env.reason in ('stalled','time_limit'):
                next_wall_index = order.index(name) + 1
                if next_wall_index < len(order):
                    print(f"[wall-incomplete] {name} ended with {env.reason}; "
                          f"will safely retract and continue to {order[next_wall_index]}.",
                          flush=True)
                else:
                    print(f"[wall-incomplete] {name} ended with {env.reason}; "
                          "no requested walls remain.",flush=True)
            elif env.reason != 'success':
                raise RuntimeError(f"Wall {name} stopped: {env.reason}; see summary.json")
            previous_wall=wall
        if args.gui: scene.flush_textures()
        if args.gui and args.hold_gui:
            while p.isConnected(scene.client):
                time.sleep(.03)
                scene._camera_keys()
    except (Exception,KeyboardInterrupt) as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
        if env is not None and env.wall is not None and (not results or results[-1]['wall']!=env.wall.name):
            results.append({'wall':env.wall.name,'steps':env.steps,'reason':'error',
                            'target_coverage':env.area/env.target_area})
        raise
    finally:
        try:
            coverage_report=print_room_coverage_summary(
                scene,wall_maps,order,path_tracker,time.perf_counter()-started)
            print('[planner-timing] '+json.dumps({
                'stage3_goal_planner':{
                    'method':'Stage3FullWallSweep.action',
                    'calls':0 if env is None else env.goal_planner_calls,
                    'total_seconds':0. if env is None else env.goal_planner_seconds,
                    'mean_seconds':(
                        env.goal_planner_seconds/env.goal_planner_calls
                        if env is not None and env.goal_planner_calls else 0.),
                },
                'motion_planner':(
                    planner.planning_timing() if planner is not None else {
                        'calls':0,'total_seconds':0.,'mean_seconds':0.,
                        'top_level_methods':{}}),
            }),flush=True)
            save(coverage_report)
        finally: scene.close()


if __name__=='__main__':
    main()
