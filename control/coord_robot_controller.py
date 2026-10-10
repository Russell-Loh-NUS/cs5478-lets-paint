"""Executor variant for the continuous coordinate-based environment: paints via real roller
contact on every physics step (not a single touch-check at the end of a move), and determines
success by reading the wall's own paint canvas back, since marking is no longer a separate flag.
"""
import math

from control.robot_controller import RobotController


class CoordRobotController(RobotController):
    # Jumps farther than this count as a "new stroke", not a continuation of the current one.
    ADJACENT_JUMP_THRESHOLD = 0.5
    # How far to pull the roller off the wall before a non-adjacent jump, so the trajectory
    # between touches doesn't stay close enough to the wall to keep dragging paint across it.
    RETRACT_DISTANCE = 0.25

    def __init__(self, robot, motion_planner, wall_map, path_tracker, coord_env,
                 joint_tolerance=0.01, max_steps_per_waypoint=200):
        super().__init__(robot, motion_planner, wall_detector=None, wall_map=wall_map,
                          path_tracker=path_tracker, joint_tolerance=joint_tolerance,
                          max_steps_per_waypoint=max_steps_per_waypoint)
        self.coord_env = coord_env
        self._last_paint_position = None

    def _execute_trajectory(self, trajectory):
        last = len(trajectory) - 1
        for i, waypoint in enumerate(trajectory):
            self.robot.set_joint_targets(waypoint)
            # Intermediate waypoints are just interpolation guides -- only the final one needs
            # the tight tolerance (see RobotController._execute_trajectory).
            tolerance = self.joint_tolerance if i == last else self.joint_tolerance * 5
            best_error, stale_steps = float("inf"), 0
            for _ in range(self.max_steps_per_waypoint):
                # env.step() (not a raw stepSimulation) so GUI texture refresh/camera panning
                # keep working live while the route executes.
                self.coord_env.step()
                axis_a, axis_b = self.robot.get_roller_axis_segment()
                self.coord_env.paint_contact(self.robot.id, axis_a, axis_b, self.robot.roller_link_index,
                                             self.robot.ROLLER_RADIUS)
                current = self.robot.get_joint_states()
                error = max(abs(c - w) for c, w in zip(current, waypoint))
                if error <= tolerance:
                    break
                # Some poses leave a genuine steady-state residual (dynamic coupling, e.g. the
                # yaw joint under load) that no amount of extra waiting resolves -- once the
                # error stops improving, stop burning the rest of the step budget.
                if error < best_error - 1e-4:
                    best_error, stale_steps = error, 0
                else:
                    stale_steps += 1
                    if stale_steps >= 20:
                        break

    def paint_target(self, target):
        # The roller stays in real contact with the wall throughout a move (that's how
        # continuous painting works), so a travel-optimized order that jumps between distant,
        # non-adjacent cells would otherwise drag a continuous paint trail across every cell
        # in between -- not just at the intended targets. Retract off the wall first for any
        # jump that isn't a short, adjacent continuation of the current stroke.
        if (self._last_paint_position is not None and
                math.dist(self._last_paint_position, target.position) > self.ADJACENT_JUMP_THRESHOLD):
            normal = self.wall_map.wall.normal
            retract_position = [target.position[i] + normal[i] * self.RETRACT_DISTANCE for i in range(3)]
            self.move_to(retract_position, target.orientation)
        self.move_to(target.position, target.orientation)
        self._last_paint_position = target.position
        if self.wall_map.is_painted(target.row, target.col):
            return True
        print(f"[CoordRobotController] Cell ({target.row},{target.col}) not sufficiently "
              "painted yet; will retry.")
        return False
