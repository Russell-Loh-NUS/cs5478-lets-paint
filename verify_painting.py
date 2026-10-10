"""Run from the project root: python verify_painting.py

Uses real PyBullet DIRECT physics, no GUI. Fails if a target is missed, the
roller hits the floor, or a cell is not painted. Allow a few minutes.
"""
import math

import pybullet as p

from simulation.paint_environment import PaintEnvironment, PaintConfig
from simulation.robot import Robot
from simulation.paint_can import PaintCan
from mapping.coord_wall_map import CoordWallMap
from planning.motion_planner import MotionPlanner, MotionPlanningError, CollisionQueryError
from planning.coverage_planner import CoverageTarget
from control.coord_robot_controller import CoordRobotController
from evaluation.path_length import PathLengthTracker


def verify_can_solidity(env, can):
    """Use real moving probes to verify that the can wall and bottom block motion."""
    for body in can.collision_body_ids:
        assert p.getCollisionShapeData(body, -1), 'Can body has no collision shape'
    radius = 0.015
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    probe = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=shape,
                             basePosition=[can.position[0] + .20, can.position[1], .14])
    try:
        p.setGravity(0, 0, 0)
        p.resetBaseVelocity(probe, linearVelocity=[-.5, 0, 0])
        hit_side = False
        for _ in range(180):
            env.step()
            hit_side |= bool(p.getContactPoints(probe, can.body_id))
        position = p.getBasePositionAndOrientation(probe)[0]
        assert hit_side and position[0] > can.position[0] + can.inner_radius, 'Probe passed through can side'

        p.resetBasePositionAndOrientation(probe, [can.position[0], can.position[1], .20], [0, 0, 0, 1])
        p.resetBaseVelocity(probe, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.setGravity(0, 0, -9.81)
        hit_bottom = False
        for _ in range(180):
            env.step()
            hit_bottom |= bool(p.getContactPoints(probe, can.bottom_id))
        position = p.getBasePositionAndOrientation(probe)[0]
        assert hit_bottom, 'Probe did not contact solid can bottom'
        assert position[2] >= can.position[2] + can.wall_thickness + radius - .002, 'Probe passed through bottom'
        print('PASS: can side blocks a moving probe; solid bottom supports a falling probe')
    finally:
        p.removeBody(probe)
        p.setGravity(0, 0, -9.81)


def verify_query_recovery(planner):
    """Inject the reported None result and check recovery; never assume free space."""
    closest, contacts = p.getClosestPoints, p.getContactPoints
    try:
        p.getClosestPoints = lambda *args, **kwargs: None
        assert not planner.current_collision(), 'Expected collision-free home pose'
        p.getContactPoints = lambda *args, **kwargs: None
        try:
            planner.current_collision()
        except CollisionQueryError:
            pass
        else:
            raise AssertionError('Missing collision data was treated as free space')
        print('PASS: None query recovered; unavailable collision data rejected')
    finally:
        p.getClosestPoints, p.getContactPoints = closest, contacts


def verify_robot_can_collision(planner, can):
    reference = planner.robot.get_joint_states()
    through_side = can.get_dip_position()
    through_side[0] += can.inner_radius + can.wall_thickness / 2
    candidates = planner._candidate_configurations(through_side, reference)
    assert candidates and planner._predicted_collision(candidates[0]), 'Roller/can-side overlap was accepted'
    chassis_in_can = list(reference)
    chassis_in_can[:2] = can.position[:2]
    assert planner._predicted_collision(chassis_in_can), 'Chassis/can overlap was accepted'
    inside = can.get_dip_position()
    candidates = planner._candidate_configurations(inside, reference)
    assert any(planner._valid_pose(q, inside, None) for q in candidates), 'Open can interior is blocked'
    assert all(abs(a-b) < 1e-9 for a,b in zip(reference, planner.robot.get_joint_states()))
    print('PASS: roller/chassis collisions detected; hollow interior remains reachable')


def verify_west_transit(planner, can):
    """Regress the yaw-stop failure on departure from the north-side dip pose.

    This checks planning against real collision geometry; the ordinary wall
    tests below separately execute movements under physics.
    """
    saved = planner._save_joints()
    try:
        above = can.get_approach_position()
        reference = [.6, 2.28, -.92, .53, .255, 0, -.53]
        candidates = planner._candidate_configurations(above, reference)
        seed = next((q for q in candidates if planner._valid_pose(q, above, None)), None)
        assert seed is not None, 'North-side can departure pose unavailable'
        planner._apply_joints(seed)
        target = [.309, 2.93, above[2]]
        trajectory = planner.plan_transit_motion(target)
        assert trajectory, 'West-corner detour is empty'
        assert all(abs(a-b) < 1e-9 for a,b in zip(seed, planner.robot.get_joint_states())), 'Planning moved the robot'
        for q in trajectory:
            assert all(lo <= v <= hi for v,lo,hi in zip(q, planner.robot.lower_limits, planner.robot.upper_limits)), 'Detour exceeds joint limits'
            assert not planner._predicted_collision(q), 'Detour collides'
        assert planner._valid_pose(trajectory[-1], target, None), 'Detour misses west approach'
        print('PASS: north-side can departure to west corner; checked yaw-stop detour')
    finally:
        planner._restore_joints(saved)


def verify_west_refill(planner, can, env):
    """Execute the previously blocked west-row-2 refill, not only a corner dip."""
    robot = planner.robot
    saved = planner._save_joints()
    # Valid retracted pose whose straight refill path fails near
    # [0.394, 2.369, 0.450] and whose simple home detours intersect the can.
    seed = [.8903372578348001, 1.8796627421651997, 2.356194490192345,
            .6018669652432583, .11141846432483216, 0., -.6018669652432584]
    try:
        assert not planner._predicted_collision(seed), 'West refill fixture is in collision'
        robot.set_joint_targets(seed, instant=True)
        target = [can.position[0], can.position[1], .45]
        trajectory = planner.plan_transit_motion(target)
        assert all(abs(a-b) < 1e-9 for a,b in zip(seed,robot.get_joint_states())), 'Transit planning moved robot'
        for q in trajectory:
            assert all(lo <= v <= hi for v,lo,hi in zip(q,robot.lower_limits,robot.upper_limits)), 'Search exceeds joint limits'
            assert not planner._predicted_collision(q), 'Search path collides'
        controller = CoordRobotController(robot, planner, None, PathLengthTracker(), env)
        assert controller._execute_trajectory(trajectory), 'West refill detour failed during physics execution'
        assert math.dist(robot.get_roller_position(),target) <= controller.POSITION_TOLERANCE
        assert controller.ensure_dipped(can), 'West refill did not reach can interior'
        print('PASS: blocked west refill detour executed and roller redipped')
    finally:
        planner._restore_joints(saved)
        robot.set_joint_targets([position for position,velocity in saved])
        robot.set_roller_color([1., .10, .10, 1.])


def main():
    env = PaintEnvironment(gui=False, paint=PaintConfig(patch_width=2 * Robot.ROLLER_RADIUS))
    try:
        can = PaintCan(position=(1.2, 1.5, 0))
        verify_can_solidity(env, can)
        robot = Robot(start_xy=(2, 1.5), max_lift=2.6,
                      x_range=(0.6, 3.4), y_range=(0.6, 2.4))
        planner = MotionPlanner(robot, [w.body_id for w in env.walls] + [env.floor_id])
        planner.configure_environment(env)
        planner.register_paint_can(can)
        verify_query_recovery(planner)
        verify_robot_can_collision(planner, can)
        verify_west_transit(planner, can)
        verify_west_refill(planner, can, env)
        tracker = PathLengthTracker()
        for name in ('south', 'east', 'north', 'west'):
            wall = env.walls_by_name[name]
            wm = CoordWallMap(wall, 2 * Robot.ROLLER_RADIUS, Robot.ROLLER_LENGTH)
            controller = CoordRobotController(robot, planner, wm, tracker, env)
            assert controller.ensure_dipped(can), f'{name}: initial dip failed'
            cells = [(0, 0), (0, 1), (0, wm.cols - 1), (1, wm.cols - 1),
                     (7, wm.cols - 1), (wm.rows - 1, 0), (wm.rows - 1, wm.cols - 1)]
            for row, col in cells:
                if wm.is_obstacle(row, col):
                    continue
                if controller.needs_redip():
                    assert controller.ensure_dipped(can), f'{name}: redip failed'
                target = CoverageTarget(row, col, wm.touch_point(row, col, Robot.ROLLER_RADIUS - .001), 'vertical')
                assert controller.paint_target(target), f'{name}: cell {(row, col)} failed'
                assert math.dist(robot.get_roller_position(), target.position) <= controller.POSITION_TOLERANCE
                contacts = p.getContactPoints(robot.id, env.floor_id, linkIndexA=robot.roller_link_index)
                assert not contacts, f'{name}: roller touched the floor'
            assert controller.ensure_dipped(can), f'{name}: final redip failed'
            print(f'PASS: {name} low/high/corner cells and refill')
        before = robot.get_joint_states()
        for target in ([2, 1, .02], [2, 1, 4]):
            try:
                planner.plan_motion(target)
            except MotionPlanningError:
                pass
            else:
                raise AssertionError(f'Invalid target accepted: {target}')
            assert all(abs(a-b) < 1e-9 for a,b in zip(before, robot.get_joint_states()))
        print('PASS: invalid targets rejected without moving the robot')
    finally:
        env.close()


if __name__ == '__main__':
    main()
