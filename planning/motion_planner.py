"""Motion planner: IK -> joint configuration -> collision checking -> joint trajectory."""
import math

import pybullet as p

# Roller spin angle (joint_roller_orient) for each mode: 0 = vertical (left/right
# sweeps), +90deg = horizontal (up/down sweeps).
ROLLER_ANGLE = {
    "vertical": 0.0,
    "horizontal": math.pi / 2.0,
}


class MotionPlanner:
    def __init__(self, robot, collision_bodies, steps_per_segment=15):
        self.robot = robot
        self.collision_bodies = collision_bodies
        self.steps_per_segment = steps_per_segment

    def plan_motion(self, target_position, orientation_mode="vertical"):
        # Position-only IK: solving for position+orientation together over this chain forces
        # the shoulder/elbow angles to always cancel out (to hold the roller's world
        # orientation fixed), which severely limits reachability. We instead let IK freely
        # choose an arm pose that reaches the target position, then set the roller's own
        # spin joint directly to express vertical/horizontal mode.
        # Seed the null-space rest pose with the CURRENT configuration (not a fixed pose) so
        # IK prefers the solution closest to where the robot already is, instead of always
        # pulling the redundant joints back toward a fixed pose between every single move.
        ik_solution = p.calculateInverseKinematics(
            self.robot.id,
            self.robot.roller_link_index,
            target_position,
            lowerLimits=self.robot.lower_limits,
            upperLimits=self.robot.upper_limits,
            jointRanges=self.robot.joint_ranges,
            restPoses=self.robot.get_joint_states(),
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        joint_targets = self._clamp(list(ik_solution[: len(self.robot.joint_indices)]))
        # Override the redundant last joint to express the requested roller mode.
        joint_targets[-1] = ROLLER_ANGLE.get(orientation_mode, ROLLER_ANGLE["vertical"])

        if self._predicted_collision(joint_targets):
            print("[MotionPlanner] Warning: planned configuration may collide with the environment.")

        current = self.robot.get_joint_states()
        return self._interpolate(current, joint_targets)

    def plan_to_joint_configuration(self, target_joints):
        """Plan a trajectory straight to a known joint configuration (e.g. sending the robot home)."""
        joint_targets = self._clamp(list(target_joints))
        if self._predicted_collision(joint_targets):
            print("[MotionPlanner] Warning: planned configuration may collide with the environment.")
        current = self.robot.get_joint_states()
        return self._interpolate(current, joint_targets)

    def _clamp(self, joints):
        return [max(lo, min(hi, j)) for j, lo, hi in zip(joints, self.robot.lower_limits, self.robot.upper_limits)]

    def _interpolate(self, start, end):
        trajectory = []
        for step in range(1, self.steps_per_segment + 1):
            t = step / self.steps_per_segment
            trajectory.append([s + (e - s) * t for s, e in zip(start, end)])
        return trajectory

    def _predicted_collision(self, joint_targets):
        """Temporarily apply the candidate configuration and check arm/mast links for collisions."""
        saved = self.robot.get_joint_states()
        for idx, pos in zip(self.robot.joint_indices, joint_targets):
            p.resetJointState(self.robot.id, idx, pos)
        p.performCollisionDetection()

        collided = False
        # Exclude the roller link itself, which is expected to approach/touch the wall.
        # distance=0.0 means only true penetration counts, not merely "close" (the forearm
        # legitimately swings near the wall while the roller paints).
        links_to_check = self.robot.joint_indices[:-1]
        for link_idx in links_to_check:
            for body in self.collision_bodies:
                points = p.getClosestPoints(self.robot.id, body, distance=0.0, linkIndexA=link_idx)
                if any(pt[8] < -0.005 for pt in points):
                    collided = True
                    break
            if collided:
                break

        for idx, pos in zip(self.robot.joint_indices, saved):
            p.resetJointState(self.robot.id, idx, pos)
        return collided