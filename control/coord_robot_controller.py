"""Execute verified motions and paint only while the loaded roller touches the active wall."""
import math

import pybullet as p

from control.robot_controller import RobotController
from planning.motion_planner import MotionPlanningError, CollisionQueryError


class CoordRobotController(RobotController):
    ADJACENT_JUMP_THRESHOLD = 0.5
    RETRACT_DISTANCE = 0.25
    TRANSIT_HEIGHT = 0.40
    POSITION_TOLERANCE = 0.012
    # Contact forces can leave a small joint residual even after the tool arrives.
    # Accept that only when the planned tool pose and actual collision checks agree.
    SETTLED_POSITION_TOLERANCE = 0.003
    SETTLED_AXIS_TOLERANCE = math.radians(2.0)
    JOINT_LIMIT_TOLERANCE = 0.003

    def __init__(self, robot, motion_planner, wall_map, path_tracker, coord_env,
                 joint_tolerance=0.002, max_steps_per_waypoint=200):
        super().__init__(robot, motion_planner, wall_detector=None, wall_map=wall_map,
                         path_tracker=path_tracker, joint_tolerance=joint_tolerance,
                         max_steps_per_waypoint=max_steps_per_waypoint)
        self.coord_env = coord_env
        self.motion_planner.configure_environment(coord_env)
        self._last_paint_position = None
        self._painting = False
        self._loaded = False
        self._allowed_contact_body = None
        self._diagnosed_cells = set()

    def _lift_stroke(self):
        for wall in self.coord_env.walls:
            wall.canvas.lift()

    def _paint_active_wall(self):
        if not self._painting or not self._loaded or self.wall_map is None:
            self._lift_stroke()
            return
        wall = self.wall_map.wall
        a, b = self.robot.get_roller_axis_segment()
        contacts = p.getContactPoints(self.robot.id, wall.body_id,
                                      linkIndexA=self.robot.roller_link_index)
        reach = self.robot.ROLLER_RADIUS + self.coord_env.paint.end_tolerance
        valid_ends = all(-self.motion_planner.WALL_PENETRATION <= wall.signed_distance(pt) <= reach
                         for pt in (a, b))
        if contacts and valid_ends:
            wall.canvas.stroke(wall.world_to_uv(a), wall.world_to_uv(b),
                               self.coord_env.paint.patch_width)
        else:
            wall.canvas.lift()

    def _step(self):
        before = self.robot.get_roller_position()
        safe_joints = self.robot.get_joint_states()
        self.coord_env.step()
        after = self.robot.get_roller_position()
        if self._painting:
            # Only count motion while the roller is actively painting against the wall.
            # Transit, retract, and approach moves should not burn the paint-distance budget.
            self.path_tracker.update_paint_distance(before, after)
        else:
            self.path_tracker.update(before, after)
        try:
            collision = self.motion_planner.current_collision(self._allowed_contact_body, executing=True)
        except CollisionQueryError as exc:
            if p.isConnected():
                self.robot.set_joint_targets(safe_joints, instant=True)
            self._lift_stroke()
            print(f"[CoordRobotController] Stopped: {exc}")
            return False
        if collision:
            # Simulation recovery: undo this one physics step, hold, and report failure.
            self.robot.set_joint_targets(safe_joints, instant=True)
            self._lift_stroke()
            print("[CoordRobotController] Stopped: unexpected collision or insufficient floor clearance.")
            return False
        self._paint_active_wall()
        return True

    def _execute_trajectory(self, trajectory):
        if not trajectory:
            return False
        for i, waypoint in enumerate(trajectory):
            self.robot.set_joint_targets(waypoint)
            tolerance = self.joint_tolerance * (1 if i == len(trajectory) - 1 else 2)
            best_error, stale_steps = float("inf"), 0
            reached = False
            for _ in range(self.max_steps_per_waypoint):
                if not self._step():
                    return False
                error = max(abs(c - w) for c, w in zip(self.robot.get_joint_states(), waypoint))
                if error <= tolerance:
                    reached = True
                    break
                if error < best_error - 1e-6:
                    best_error, stale_steps = error, 0
                else:
                    stale_steps += 1
                    if stale_steps >= 40:
                        break
            if not reached:
                try:
                    tool_reached, tool_error = self._settled_tool_pose_reached(waypoint)
                except CollisionQueryError as exc:
                    self.robot.set_joint_targets(self.robot.get_joint_states())
                    self._lift_stroke()
                    print(f"[CoordRobotController] Stopped: {exc}")
                    return False
                if tool_reached:
                    print(f"[CoordRobotController] Tool pose reached "
                          f"(position error {tool_error:.4f} m; "
                          f"residual joint error {error:.4f}).")
                    continue
                self.robot.set_joint_targets(self.robot.get_joint_states())
                self._lift_stroke()
                print(f"[CoordRobotController] Motion stalled (joint error {error:.4f}); move failed.")
                return False
        return True

    def _settled_tool_pose_reached(self, waypoint):
        """Accept a reached tool pose under contact, never an unsafe/missed pose.

        Compare with this waypoint's FK, not the final target of the whole move.
        This is used only after the normal joint controller stalls or times out.
        Temporary FK evaluation restores joint positions AND velocities, without
        changing the motor targets that maintain wall contact.
        """
        current = self.robot.get_joint_states()
        # This arm is redundant: two valid joint configurations can give the
        # same roller pose. A maximum joint-error cutoff caused false failures
        # at 0.0161 even with only 0.0005 m of tool error. Check physical joint
        # limits instead; tool position, axis and collisions decide acceptance.
        margin = self.JOINT_LIMIT_TOLERANCE
        if not all(math.isfinite(q) and lo - margin <= q <= hi + margin for q, lo, hi in
                   zip(current, self.robot.lower_limits, self.robot.upper_limits)):
            return False, float("inf")
        actual_position = self.robot.get_roller_position()
        actual_a, actual_b = self.robot.get_roller_axis_segment()
        saved = self.motion_planner._save_joints()
        try:
            self.motion_planner._apply_joints(waypoint)
            expected_position = self.robot.get_roller_position()
            expected_a, expected_b = self.robot.get_roller_axis_segment()
        finally:
            self.motion_planner._restore_joints(saved)

        error = math.dist(actual_position, expected_position)
        actual_axis = [a - b for a, b in zip(actual_a, actual_b)]
        expected_axis = [a - b for a, b in zip(expected_a, expected_b)]
        length_product = math.sqrt(sum(v*v for v in actual_axis) *
                                   sum(v*v for v in expected_axis))
        if length_product <= 1e-12:
            return False, error
        # Cylinder axes are equivalent under a 180-degree reversal.
        alignment = abs(sum(a*b for a, b in zip(actual_axis, expected_axis))) / length_product
        reached = (error <= self.SETTLED_POSITION_TOLERANCE and
                   alignment >= math.cos(self.SETTLED_AXIS_TOLERANCE) and
                   not self.motion_planner.current_collision(self._allowed_contact_body, executing=True))
        return reached, error

    def move_to(self, target_position, orientation_mode="vertical", allowed_contact_body=None,
                *, transit=False, retract=False):
        if transit and (self._painting or allowed_contact_body is not None or
                        orientation_mode != "vertical"):
            raise ValueError("Transit detours require paint-off, contact-free vertical travel")
        if retract and (transit or self._painting or orientation_mode != "vertical"):
            raise ValueError("Wall retreat detours require paint-off vertical travel")
        self._allowed_contact_body = allowed_contact_body
        try:
            if retract:
                trajectory = self.motion_planner.plan_retraction_motion(
                    target_position, allowed_contact_body)
            elif transit:
                trajectory = self.motion_planner.plan_transit_motion(target_position)
            else:
                trajectory = self.motion_planner.plan_motion(
                    target_position, orientation_mode, allowed_contact_body=allowed_contact_body)
        except MotionPlanningError as exc:
            print(f"[CoordRobotController] Move rejected: {exc}")
            self._last_paint_position = None
            return False
        success = self._execute_trajectory(trajectory)
        error = math.dist(self.robot.get_roller_position(), target_position)
        if not success:
            self._last_paint_position = None
            print(f"[CoordRobotController] Motion incomplete; tool position error {error:.4f} m.")
            return False
        if error > self.POSITION_TOLERANCE:
            self._last_paint_position = None
            print(f"[CoordRobotController] Target missed by {error:.3f} m.")
            return False
        return True

    def _retract(self):
        """Leave the CURRENT contact point before travelling toward the next target."""
        self._painting = False
        self._lift_stroke()
        position = self.robot.get_roller_position()
        retreat = list(position)
        touching = []
        for wall in self.coord_env.walls:
            distance = wall.signed_distance(position)
            if p.getContactPoints(self.robot.id, wall.body_id,
                                  linkIndexA=self.robot.roller_link_index):
                touching.append(wall.body_id)
            gap = self.robot.ROLLER_RADIUS + self.RETRACT_DISTANCE - distance
            if gap > 0:
                retreat = [v + float(n) * gap for v, n in zip(retreat, wall.normal)]
        if len(touching) > 1:
            print("[CoordRobotController] Cannot retract: roller touches multiple walls.")
            return False
        if math.dist(position, retreat) < 0.001:
            return True
        allowed = touching[0] if touching else None
        return self.move_to(retreat, allowed_contact_body=allowed, retract=True)

    def _travel_to(self, target_position):
        """Paint-off travel: retract, lift, translate, and lower."""
        self._last_paint_position = None
        self._painting = False
        self._lift_stroke()
        if not self._lift_out_of_can():
            return False
        if not self._retract():
            return False
        start = self.robot.get_roller_position()
        target = list(target_position)
        height = max(self.TRANSIT_HEIGHT, start[2], target[2])
        for index, waypoint in enumerate(([start[0], start[1], height],
                                          [target[0], target[1], height], target)):
            if math.dist(self.robot.get_roller_position(), waypoint) > 0.001:
                # Only lateral paint-off travel may detour. Can entry/exit and
                # the final descent retain their strictly vertical paths.
                if not self.move_to(waypoint, transit=(index == 1)):
                    return False
        return True

    def _lift_out_of_can(self):
        """A dipped roller must clear the rim before ANY lateral travel."""
        for can in self.motion_planner.paint_cans:
            current = self.robot.get_roller_position()
            above = can.get_approach_position(self.robot.ROLLER_LENGTH, self.robot.ROLLER_RADIUS)
            radial = math.hypot(current[0] - can.position[0], current[1] - can.position[1])
            if (radial < can.outer_radius + self.robot.ROLLER_RADIUS + 0.02 and
                    current[2] < above[2] - 0.001):
                if not self.move_to([current[0], current[1], above[2]]):
                    return False
        return True

    def dip_in_paint(self, paint_can):
        self.motion_planner.register_paint_can(paint_can)
        # A scheduled dip can immediately follow the actual-distance safety dip.
        # If still loaded and inside this can, do not repeat the entire operation.
        if self._loaded and self.path_tracker.distance_since_dip() < 0.001:
            a, b = self.robot.get_roller_axis_segment()
            if paint_can.try_dip(a, b, self.robot.ROLLER_RADIUS):
                return True
        self._loaded = False
        above = paint_can.get_approach_position(self.robot.ROLLER_LENGTH, self.robot.ROLLER_RADIUS)
        if not self._travel_to(above):
            return False
        # Enter vertically through the open top; do not intersect a solid side wall.
        if not self.move_to(paint_can.get_dip_position()):
            return False
        a, b = self.robot.get_roller_axis_segment()
        if not paint_can.try_dip(a, b, self.robot.ROLLER_RADIUS):
            print("[CoordRobotController] Roller did not reach the paint can.")
            return False
        self._loaded = True
        self.robot.set_roller_color([0.0, 0.0, 1.0, 1.0])
        self.path_tracker.reset_dip_counter()
        print("[RobotController] Roller dipped in paint.")
        return True

    def paint_target(self, target):
        if not self._loaded:
            print("[CoordRobotController] Painting requires a successful dip.")
            return False
        wall = self.wall_map.wall
        adjacent = (self._last_paint_position is not None and
                    math.dist(self._last_paint_position, target.position) <= self.ADJACENT_JUMP_THRESHOLD)
        if not adjacent:
            approach = [float(v + n * self.RETRACT_DISTANCE)
                        for v, n in zip(target.position, wall.normal)]
            if not self._travel_to(approach):
                return False
        self._painting = True
        try:
            if not self.move_to(target.position, target.orientation, wall.body_id):
                return False
            # Hold briefly to establish wall contact and settle the 1 mm contact offset.
            for _ in range(40):
                if not self._step():
                    self._last_paint_position = None
                    return False
            actual = self.robot.get_roller_position()
            a, b = self.robot.get_roller_axis_segment()
            error = math.dist(actual, target.position)
            verticality = abs(a[2] - b[2]) / self.robot.ROLLER_LENGTH
            contacts = p.getContactPoints(self.robot.id, wall.body_id,
                                          linkIndexA=self.robot.roller_link_index)
            reached = error <= self.POSITION_TOLERANCE and verticality >= 0.995 and bool(contacts)
            cell = (target.row, target.col)
            if cell not in self._diagnosed_cells and len(self._diagnosed_cells) < 3:
                self._diagnosed_cells.add(cell)
                print(f"[DIAG] cell={cell} target={[round(v, 3) for v in target.position]} "
                      f"actual={[round(v, 3) for v in actual]} position_error={error:.3f}m "
                      f"verticality={verticality:.4f} wall_contact={bool(contacts)}")
            self._last_paint_position = actual if reached else None
            cell_fraction = self.wall_map.cell_painted_fraction(target.row, target.col)
            success = reached and cell_fraction >= 0.80
            if not success:
                print(f"[CoordRobotController] Cell {cell} not reached/painted enough; "
                      f"fraction={cell_fraction:.2%}, target_threshold=80%.")
            return success
        finally:
            self._painting = False
            self._lift_stroke()

    def return_to_center(self):
        self._loaded = False
        self._painting = False
        if not self._lift_out_of_can():
            return False
        if not self._retract():
            return False
        self._allowed_contact_body = None
        try:
            trajectory = self.motion_planner.plan_to_joint_configuration(self.robot.home_joints)
        except MotionPlanningError as exc:
            print(f"[CoordRobotController] Return home rejected: {exc}")
            return False
        return self._execute_trajectory(trajectory)
