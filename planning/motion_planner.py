"""Motion planner: IK -> joint configuration -> collision checking -> joint trajectory."""
import math

import pybullet as p


class MotionPlanner:
    def __init__(self, robot, collision_bodies, steps_per_segment=15):
        self.robot = robot
        self.collision_bodies = collision_bodies
        self.steps_per_segment = steps_per_segment

    def plan_motion(self, target_position, orientation_mode="vertical"):
        # Position-only IK: solving for position+orientation together over this chain forces
        # the shoulder/elbow angles to always cancel out (to hold the roller's world
        # orientation fixed), which severely limits reachability. We instead let IK freely
        # choose an arm pose that reaches the target position, then separately solve for the
        # roller's own spin joint -- the one remaining DOF -- so its cylindrical AXIS actually
        # ends up vertical (flush barrel contact), not just applying a fixed angle that ignores
        # whatever yaw/shoulder/elbow pose IK happened to pick for reachability.
        joint_targets = self._solve_ik(target_position)
        joint_targets[-1] = self._solve_roller_orient(joint_targets, orientation_mode)

        current = self.robot.get_joint_states()
        return self._interpolate(current, joint_targets)

    def _solve_roller_orient(self, joint_targets, orientation_mode):
        """The roller's cylindrical axis starts aligned with the forearm's own local Z/X axes
        (the roller_orient joint only rotates it about the forearm's local Y) -- so a FIXED
        spin angle only produces a vertical axis by coincidence, since yaw/shoulder/elbow tilt
        the forearm differently for every target. Instead, read the forearm's actual world
        orientation for this specific pose and solve for the spin angle that rotates the axis
        as close to vertical (or horizontal, 90 deg off) as this one remaining DOF allows."""
        forearm_idx = self.robot.joint_indices[self.robot.joint_names.index("joint_elbow")]
        saved = self.robot.get_joint_states()
        for idx, pos in zip(self.robot.joint_indices, joint_targets):
            p.resetJointState(self.robot.id, idx, pos)
        orn = p.getLinkState(self.robot.id, forearm_idx, computeForwardKinematics=True)[1]
        rot = p.getMatrixFromQuaternion(orn)
        forearm_x = (rot[0], rot[3], rot[6])
        forearm_z = (rot[2], rot[5], rot[8])
        for idx, pos in zip(self.robot.joint_indices, saved):
            p.resetJointState(self.robot.id, idx, pos)

        # Rotating the roller's local Z (= forearm's local Z before this joint) about the
        # forearm's local Y by angle t traces cos(t)*forearm_z + sin(t)*forearm_x -- maximize
        # its dot with world Z (vertical) by picking t = atan2(forearm_x.z, forearm_z.z).
        theta_vertical = math.atan2(forearm_x[2], forearm_z[2])
        theta = theta_vertical if orientation_mode == "vertical" else theta_vertical + math.pi / 2.0
        theta = (theta + math.pi) % (2 * math.pi) - math.pi  # wrap before clamping to limits
        lo, hi = self.robot.lower_limits[-1], self.robot.upper_limits[-1]
        return max(lo, min(hi, theta))

    def _solve_ik(self, target_position, position_tolerance=0.02):
        # Seed the null-space rest pose with the CURRENT configuration (not a fixed pose) so
        # IK prefers the solution closest to where the robot already is, instead of always
        # pulling the redundant joints back toward a fixed pose between every single move.
        # But if the arm is currently straight (shoulder/elbow ~ the rest-pose bias), the
        # solver can get stuck trying to reach purely via a prismatic joint hitting its limit
        # (e.g. lift -> 0) without ever exploring bending the arm -- so also try a few seeds
        # with the arm nudged into different bends, PLUS seeds that reposition the base near
        # the target (all arm-only seeds keep the base fixed, so the null-space solver would
        # otherwise never discover "drive closer" and instead strains the shoulder/elbow past
        # their limits trying to reach far-away cells without moving). Among everything that
        # actually reaches the target once clamped, prefer whichever is also collision-free
        # (the robot must not collide with the wall; a close-but-colliding solution is only
        # used as a last resort).
        current = self.robot.get_joint_states()
        shoulder_i = self.robot.joint_names.index("joint_shoulder")
        elbow_i = self.robot.joint_names.index("joint_elbow")
        base_x_i = self.robot.joint_names.index("joint_base_x")
        base_y_i = self.robot.joint_names.index("joint_base_y")

        def nudge(d_shoulder, d_elbow):
            seed = list(current)
            seed[shoulder_i] += d_shoulder
            seed[elbow_i] += d_elbow
            return seed

        def base_near_target(d_shoulder=0.0, d_elbow=0.0):
            # All of the seeds above keep the base wherever it already is and only vary the
            # arm, so the null-space solver never discovers "drive closer, then reach" --
            # it'll happily contort the arm to its limits instead of moving the base at all.
            # Explicitly seed the base near the target's x/y so IK can find (and then prefer,
            # once it's lower-error/collision-free) solutions that actually relocate it.
            seed = list(current)
            seed[base_x_i] = max(self.robot.x_range[0], min(self.robot.x_range[1], target_position[0]))
            seed[base_y_i] = max(self.robot.y_range[0], min(self.robot.y_range[1], target_position[1]))
            seed[shoulder_i] = d_shoulder
            seed[elbow_i] = d_elbow
            return seed

        seeds = [current, nudge(0.4, -0.4), nudge(-0.4, 0.4), nudge(0.6, -0.2), nudge(-0.2, 0.6),
                 base_near_target(), base_near_target(0.4, -0.4), base_near_target(-0.4, 0.4)]

        best_joints, best_error, best_collision_free, best_verticality = None, float("inf"), False, -1.0
        for seed in seeds:
            solution = p.calculateInverseKinematics(
                self.robot.id,
                self.robot.roller_link_index,
                target_position,
                lowerLimits=self.robot.lower_limits,
                upperLimits=self.robot.upper_limits,
                jointRanges=self.robot.joint_ranges,
                restPoses=seed,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )
            joints = self._clamp(list(solution[: len(self.robot.joint_indices)]))
            error = self._residual_error(joints, target_position)
            if error > position_tolerance * 10:
                continue  # nowhere near the target, not worth a collision check
            collision_free = not self._predicted_collision(joints)
            # Reaching the target position isn't the whole job: joint_roller_orient has a
            # limited range (can't spin a full half-turn), so some arm poses leave NO way to
            # get the roller's axis genuinely vertical even once it's set -- it ends up tilted,
            # painting with the end cap instead of the barrel. Once a candidate is accurate
            # enough, prefer whichever ALSO leaves the roller able to reach near-vertical.
            verticality = self._achievable_verticality(joints)
            candidate_ok = error <= position_tolerance
            best_ok = best_error <= position_tolerance
            if best_joints is None:
                better = True
            elif collision_free != best_collision_free:
                better = collision_free
            elif candidate_ok and best_ok:
                better = verticality > best_verticality
            elif candidate_ok != best_ok:
                better = candidate_ok
            else:
                better = error < best_error
            if better:
                best_joints, best_error, best_collision_free, best_verticality = joints, error, collision_free, verticality
            if best_collision_free and best_error <= position_tolerance and best_verticality > 0.9:
                break

        if best_joints is None:
            # Nothing came reasonably close in any seed; fall back to the plain current-state
            # solve so there's always something to return, even if it's a poor solution.
            solution = p.calculateInverseKinematics(
                self.robot.id, self.robot.roller_link_index, target_position,
                lowerLimits=self.robot.lower_limits, upperLimits=self.robot.upper_limits,
                jointRanges=self.robot.joint_ranges, restPoses=current,
                maxNumIterations=200, residualThreshold=1e-4,
            )
            best_joints = self._clamp(list(solution[: len(self.robot.joint_indices)]))
            best_collision_free = not self._predicted_collision(best_joints)

        if not best_collision_free:
            print("[MotionPlanner] Warning: no collision-free configuration found for this move "
                  "after trying multiple arm poses; proceeding with the closest available.")
        return best_joints

    def _residual_error(self, joints, target_position):
        """How far the roller actually ends up from the target if these joints are applied."""
        saved = self.robot.get_joint_states()
        for idx, pos in zip(self.robot.joint_indices, joints):
            p.resetJointState(self.robot.id, idx, pos)
        achieved = self.robot.get_roller_position()
        for idx, pos in zip(self.robot.joint_indices, saved):
            p.resetJointState(self.robot.id, idx, pos)
        return math.dist(achieved, target_position)

    def _achievable_verticality(self, joints):
        """Best roller-axis alignment with vertical achievable within joint_roller_orient's own
        limited range, for this particular arm pose -- a tie-breaker so IK prefers poses that
        let the roller actually end up vertical, not just ones that reach the target position
        (see _solve_roller_orient for the rotation math this mirrors)."""
        forearm_idx = self.robot.joint_indices[self.robot.joint_names.index("joint_elbow")]
        saved = self.robot.get_joint_states()
        for idx, pos in zip(self.robot.joint_indices, joints):
            p.resetJointState(self.robot.id, idx, pos)
        orn = p.getLinkState(self.robot.id, forearm_idx, computeForwardKinematics=True)[1]
        rot = p.getMatrixFromQuaternion(orn)
        a, b = rot[8], rot[6]  # forearm local Z and X axes' world-Z components
        for idx, pos in zip(self.robot.joint_indices, saved):
            p.resetJointState(self.robot.id, idx, pos)

        lo, hi = self.robot.lower_limits[-1], self.robot.upper_limits[-1]
        theta = (math.atan2(b, a) + math.pi) % (2 * math.pi) - math.pi
        theta = max(lo, min(hi, theta))
        return a * math.cos(theta) + b * math.sin(theta)

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
