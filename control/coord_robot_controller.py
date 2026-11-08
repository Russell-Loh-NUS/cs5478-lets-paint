"""Execute verified motions and paint only while the loaded roller touches the active wall."""
import math
from contextlib import contextmanager

import pybullet as p

from control.robot_controller import RobotController
from planning.motion_planner import MotionPlanningError, CollisionQueryError
from planning.motion_planner_rl import OUTBOUND, RETURN, stowed_joints, wrap_angle


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
                 joint_tolerance=0.002, max_steps_per_waypoint=200, base_navigator=None,
                 rl_fallback=True):
        super().__init__(robot, motion_planner, wall_detector=None, wall_map=wall_map,
                         path_tracker=path_tracker, joint_tolerance=joint_tolerance,
                         max_steps_per_waypoint=max_steps_per_waypoint)
        self.coord_env = coord_env
        self.motion_planner.configure_environment(coord_env)
        # Optional planning.motion_planner_rl.BaseNavigator. When set, the long base legs to the
        # paint can and back to the wall are driven by the RL policy - everything else (and any
        # failure of the policy) uses the classic planner, so None keeps the old behaviour.
        self.base_navigator = base_navigator
        # False = no safety net - a failed RL leg aborts the dip / the cell instead of continuing
        # with the classic travel, so the policy's own performance is what shows up.
        self.rl_fallback = rl_fallback
        self._resume_pending = False
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

    @contextmanager
    def _fixed_base(self):
        # Plan with the base where it is - only the arm moves (see MotionPlanner.fixed_base).
        self.motion_planner.fixed_base = True
        try:
            yield
        finally:
            self.motion_planner.fixed_base = False

    def _arm_to(self, point):
        # Arm-only move to a roller position. The arm joints come from the analytic IK for the base
        # where it is, and the move is a straight blend of the joints (collision checked), not a
        # straight line of the roller - that way it cannot be forced through the arm's dead zone
        # next to the base. Returns False, moving nothing, if no collision-free pose exists.
        mp = self.motion_planner
        self._allowed_contact_body = None
        with self._fixed_base():
            # A joint pushed onto its stop can read a hair past it (base y = 0.5999967 against a
            # limit of 0.6), and the IK rejects any pose outside the limits, so it would find no
            # arm solution at all. Solve from the base clamped into its range.
            current = [min(max(v, lo), hi) for v, lo, hi in
                       zip(self.robot.get_joint_states(), self.robot.lower_limits, self.robot.upper_limits)]
            for joints in mp._candidate_configurations(point, current):
                if not mp._valid_pose(joints, point, None):
                    continue
                # Two stages - turn to face the target with the arm still stowed (the roller stays
                # over the base, so nothing can be hit sideways), then swing the arm down/out in
                # that vertical plane. Blending everything at once sweeps the roller through
                # intermediate headings and can clip a nearby wall.
                # The stowed roller lies horizontal (it sticks out ~9 cm to each side), so it is also
                # stood upright while the arm is still stowed and kept vertical for the whole swing - 
                # otherwise it clips walls that are close to the target (corner cells).
                turned = list(current)
                turned[2] = joints[2]
                upright = list(turned)
                upright[6] = wrap_angle(-turned[3])
                swing = mp._interpolate(upright, joints)
                for q in swing:
                    q[6] = wrap_angle(-q[3])
                trajectory = mp._interpolate(current, turned) + mp._interpolate(turned, upright) + swing
                if any(mp._predicted_collision(q) for q in trajectory):
                    continue
                return self._execute_trajectory(trajectory)
        print(f"[RL motion] The arm cannot reach {[round(v, 2) for v in point]} from the base position.")
        return False

    def _stow_arm(self):
        # Bring the arm to the stowed pose (vertical, retracted, roller horizontal) with the base
        # where it is - the mirror image of _arm_to. The roller stays vertical while the arm swings
        # up and in, and is laid horizontal only at the top. Swinging up passes through the arm's
        # minimum-reach circle (~0.96 m), so a base standing 0.65 m from the wall it was painting
        # cannot do it facing that wall - the arm is first turned (still low and retracted) toward
        # open space, trying headings from "toward the middle of the room" outward.
        mp = self.motion_planner
        lower, upper = self.robot.lower_limits, self.robot.upper_limits
        current = self.robot.get_joint_states()
        # A base joint resting on its stop can read a hair past it (yaw +3.14159...), which the
        # planner rejects as an invalid configuration - clamp before planning.
        base = [min(max(v, lo), hi) for v, lo, hi in zip(current[:3], lower, upper)]
        if max(abs(a - b) for a, b in zip(current[3:], stowed_joints(*base)[3:])) < 1e-3:
            return True  # already stowed
        start = list(current)
        start[:3] = base
        start_v = list(start)
        start_v[6] = wrap_angle(-start[3])
        centre = self.base_navigator._poly.mean(axis=0)
        toward = math.atan2(centre[1] - base[1], centre[0] - base[0])
        headings = [start[2]] + [toward + k * math.pi / 6 for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6)]
        self._allowed_contact_body = None
        for heading in headings:
            yaw = min(max(wrap_angle(heading), lower[2]), upper[2])
            turned = list(start_v)
            turned[2] = yaw
            stowed = stowed_joints(base[0], base[1], yaw)
            upright = list(stowed)
            upright[6] = wrap_angle(-stowed[3])
            swing = mp._interpolate(turned, upright)
            for q in swing:
                q[6] = wrap_angle(-q[3])
            trajectory = (mp._interpolate(start, start_v) + mp._interpolate(start_v, turned) + swing +
                          mp._interpolate(upright, stowed))
            if any(mp._predicted_collision(q) for q in trajectory):
                continue
            return self._execute_trajectory(trajectory)
        print("[RL motion] Could not stow the arm: the swing up collides in every heading tried.")
        return False

    def navigate_base(self, phase, goal_xy, band):
        # Stow the arm (vertical, retracted) and let the RL policy drive the base into a distance
        # band around `goal_xy`. Returns True on arrival, with the arm stowed. On False the base
        # may be anywhere on the way, also with the arm stowed.
        nav = self.base_navigator
        if nav is None:
            return False
        nav.start(phase, goal_xy, band)
        self._last_paint_position = None
        self._painting = False
        self._lift_stroke()
        if not self._lift_out_of_can() or not self._retract():
            return False
        self._allowed_contact_body = None
        if not self._stow_arm():
            return False
        state = "success" if nav.arrived() else "continue"
        while state == "continue":
            state, pose = nav.step()
            if state != "continue":
                break
            waypoint = stowed_joints(*pose)
            # Look before moving - the policy can propose a pose that clips a wall or the can.
            # Check it against the same margin the policy was trained with, without moving.
            saved = self.motion_planner._save_joints()
            try:
                self.motion_planner._apply_joints(waypoint)
                blocked = nav._collides(nav.cfg.collision_margin)
            finally:
                self.motion_planner._restore_joints(saved)
            if blocked:
                state = "blocked"
            elif not self._execute_trajectory([waypoint]):
                state = "execution_failed"
        nav.read_pose()
        dist = nav._goal_metrics()[0]
        print(f"[RL motion] {'outbound' if phase == OUTBOUND else 'return'} {state} after {nav._steps} steps; "
              f"base at ({nav.x:.2f}, {nav.y:.2f}), {dist:.2f} m from the goal point "
              f"(band {band[0]:.2f}-{band[1]:.2f} m), yaw {math.degrees(nav.yaw):+.0f} deg.")
        return state == "success"

    def _rl_resume(self, target, wall):
        # RL return - drive into the band in front of the target, then bring the roller to the
        # approach point in front of it with the arm alone.
        nav = self.base_navigator
        if not self.navigate_base(RETURN, target.position[:2], nav.paint_band(target.position[2])):
            return False
        approach = [float(v + n * self.RETRACT_DISTANCE) for v, n in zip(target.position, wall.normal)]
        return self._arm_to(approach)

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

    def _finish_dip(self):
        self._loaded = True
        self._resume_pending = True
        self.robot.set_roller_color([0.0, 0.0, 1.0, 1.0])
        self.path_tracker.reset_dip_counter()
        print("[RobotController] Roller dipped in paint.")
        return True

    def _arm_only_dip(self, paint_can):
        # Base already in the ring around the can, arm stowed - swing the arm over the can, go down
        # through the open top, dip, and come back up. The base does not move.
        above = paint_can.get_approach_position(self.robot.ROLLER_LENGTH, self.robot.ROLLER_RADIUS)
        if not self._arm_to(above):
            return False
        with self._fixed_base():
            if not self.move_to(paint_can.get_dip_position()):
                return False
            a, b = self.robot.get_roller_axis_segment()
            if not paint_can.try_dip(a, b, self.robot.ROLLER_RADIUS):
                print("[CoordRobotController] Roller did not reach the paint can.")
                return False
            self._finish_dip()
            return self.move_to(above)  # clear the rim - the next base trip stows the arm

    def dip_in_paint(self, paint_can):
        self.motion_planner.register_paint_can(paint_can)
        # A scheduled dip can immediately follow the actual-distance safety dip.
        # If still loaded and inside this can, do not repeat the entire operation.
        if self._loaded and self.path_tracker.distance_since_dip() < 0.001:
            a, b = self.robot.get_roller_axis_segment()
            if paint_can.try_dip(a, b, self.robot.ROLLER_RADIUS):
                return True
        self._loaded = False
        self._resume_pending = False
        if self.base_navigator is not None:
            # RL positions the base in the ring around the can - the arm does the dip by itself.
            arrived = self.navigate_base(OUTBOUND, paint_can.position[:2], self.base_navigator.cfg.dip_band)
            if arrived and self._arm_only_dip(paint_can):
                return True
            if not self.rl_fallback:
                print("[RL motion] Outbound leg or arm-only dip failed and the classic fallback is "
                      "disabled; dip aborted.")
                return False
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
        return self._finish_dip()

    def paint_target(self, target):
        if not self._loaded:
            print("[CoordRobotController] Painting requires a successful dip.")
            return False
        wall = self.wall_map.wall
        adjacent = (self._last_paint_position is not None and
                    math.dist(self._last_paint_position, target.position) <= self.ADJACENT_JUMP_THRESHOLD)
        if not adjacent:
            arrived = False
            if self._resume_pending and self.base_navigator is not None:
                arrived = self._rl_resume(target, wall)
                if not arrived and not self.rl_fallback:
                    # Stay pending so the retry of this cell (or the next one) tries the RL again.
                    print("[RL motion] Return leg failed and the classic fallback is disabled; "
                          f"cell ({target.row}, {target.col}) not painted.")
                    return False
            self._resume_pending = False
            if not arrived:
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
