"""Executor: carries out planned joint trajectories and updates paint/coverage state."""
import pybullet as p


class RobotController:
    # Roller travel distance after which it's considered out of paint and needs a redip.
    REDIP_DISTANCE = 10.0
    MAX_DIP_ATTEMPTS = 5

    def __init__(self, robot, motion_planner, wall_detector, wall_map, path_tracker,
                 joint_tolerance=0.01, max_steps_per_waypoint=200):
        self.robot = robot
        self.motion_planner = motion_planner
        self.wall_detector = wall_detector
        self.wall_map = wall_map
        self.path_tracker = path_tracker
        self.joint_tolerance = joint_tolerance
        self.max_steps_per_waypoint = max_steps_per_waypoint

    def _execute_trajectory(self, trajectory):
        last = len(trajectory) - 1
        for i, waypoint in enumerate(trajectory):
            self.robot.set_joint_targets(waypoint)
            # Intermediate waypoints are just interpolation guides -- only the final one needs
            # the tight tolerance; holding every single one to it wastes steps settling exactly
            # onto points the robot is about to leave again anyway.
            tolerance = self.joint_tolerance if i == last else self.joint_tolerance * 5
            # Step until the controllers actually reach this waypoint (or give up after a cap).
            best_error, stale_steps = float("inf"), 0
            for _ in range(self.max_steps_per_waypoint):
                p.stepSimulation()
                current = self.robot.get_joint_states()
                error = max(abs(c - w) for c, w in zip(current, waypoint))
                if error <= tolerance:
                    break
                # Some poses leave a genuine steady-state residual (e.g. dynamic coupling the
                # yaw joint can't fully null out) that no amount of extra waiting resolves --
                # once the error stops improving, stop burning the rest of the step budget.
                if error < best_error - 1e-4:
                    best_error, stale_steps = error, 0
                else:
                    stale_steps += 1
                    if stale_steps >= 20:
                        break

    def move_to(self, target_position, orientation_mode="vertical"):
        start_position = self.robot.get_roller_position()
        trajectory = self.motion_planner.plan_motion(target_position, orientation_mode)
        self._execute_trajectory(trajectory)
        end_position = self.robot.get_roller_position()
        self.path_tracker.update(start_position, end_position)

    def dip_in_paint(self, paint_can):
        self.move_to(paint_can.get_dip_position(), orientation_mode="vertical")
        # Both axis endpoints, not a single point -- the whole roller must be inside the can.
        roller_a, roller_b = self.robot.get_roller_axis_segment()
        if paint_can.try_dip(roller_a, roller_b):
            self.robot.set_roller_color([0.0, 0.0, 1.0, 1.0])
            self.path_tracker.reset_dip_counter()
            print("[RobotController] Roller dipped in paint.")
            return True
        print("[RobotController] Warning: roller did not reach the paint can.")
        return False

    def ensure_dipped(self, paint_can):
        """Keep retrying the dip until it actually succeeds; the roller must not go paint without it."""
        for attempt in range(1, self.MAX_DIP_ATTEMPTS + 1):
            if self.dip_in_paint(paint_can):
                return True
            print(f"[RobotController] Retrying dip ({attempt}/{self.MAX_DIP_ATTEMPTS})...")
        print(f"[RobotController] Giving up on dipping after {self.MAX_DIP_ATTEMPTS} attempts.")
        return False

    def needs_redip(self):
        return self.path_tracker.distance_since_dip() >= self.REDIP_DISTANCE

    def paint_target(self, target):
        self.move_to(target.position, target.orientation)
        if not self.wall_detector.is_touching():
            print(f"[RobotController] Roller did not reach wall for cell ({target.row},{target.col}); will retry.")
            return False

        # IK convergence noise means the roller's raw position can round into a neighboring
        # cell, so the intended target stays the source of truth for marking success. We only
        # use the actual position to catch a genuine obstacle violation (the roller drifting
        # onto a non-paintable cell while reaching for one next to it).
        actual_row, actual_col = self.wall_map.world_to_grid(self.robot.get_roller_position())
        if (actual_row, actual_col) != (target.row, target.col) and self.wall_map.is_obstacle(actual_row, actual_col):
            self.wall_map.mark_painted(actual_row, actual_col)

        self.wall_map.mark_painted(target.row, target.col)
        return True

    def return_to_center(self):
        start_position = self.robot.get_roller_position()
        trajectory = self.motion_planner.plan_to_joint_configuration(self.robot.home_joints)
        self._execute_trajectory(trajectory)
        end_position = self.robot.get_roller_position()
        self.path_tracker.update(start_position, end_position)
