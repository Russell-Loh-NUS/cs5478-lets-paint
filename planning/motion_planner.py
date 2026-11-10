"""Collision-checked motion for simulation.robot.Robot's telescopic arm.

Joint order: base X, base Y, base yaw, shoulder pitch, extension,
wrist roll, roller pitch. The legacy joint names do not describe these roles.
"""
import math
import random

import pybullet as p


class MotionPlanningError(RuntimeError):
    """No valid path was found; never execute a fallback pose."""


class CollisionQueryError(MotionPlanningError):
    """Collision state could not be obtained; stop rather than assuming free space."""


class MotionPlanner:
    # Dimensions from the current robot.py URDF, in metres.
    SHOULDER_HEIGHT = 1.0
    SHOULDER_OFFSET = 0.02
    FIXED_REACH = 0.10 + 0.55 + 0.21
    FLOOR_CLEARANCE = 0.01
    WALL_PENETRATION = 0.004  # allow the requested 1 mm painting contact
    CARTESIAN_STEP = 0.025
    JOINT_STEP = 0.04

    def __init__(self, robot, collision_bodies, steps_per_segment=15):
        self.robot = robot
        self.collision_bodies = list(collision_bodies)
        self.steps_per_segment = steps_per_segment
        self.floor_z = 0.0
        self.walls = []
        self.paint_cans = []
        self._reported_query_fallback = False
        # Opt-in (default off) - plan with the base where it is. The arm joints still come from the
        # analytic IK (yaw, shoulder pitch, extension) and every pose is still collision checked,
        # but there is no search over other base positions and no home-pose / RRT detours. Used
        # for the redip legs, where the RL policy is responsible for placing the base.
        self.fixed_base = False

    def configure_environment(self, env):
        self.walls = list(env.walls)
        self.floor_z = p.getAABB(env.floor_id)[1][2]

    def register_paint_can(self, paint_can):
        if paint_can not in self.paint_cans:
            self.paint_cans.append(paint_can)
        for body in paint_can.collision_body_ids:
            if body not in self.collision_bodies:
                self.collision_bodies.append(body)

    def _collision_points(self, body):
        points = p.getClosestPoints(self.robot.id, body, distance=0.0)
        if points is not None:
            return points
        # Some builds return None when this query is unavailable. It is not a
        # contact list and must not be iterated or silently treated as empty.
        if not p.isConnected():
            raise CollisionQueryError("Physics connection closed during collision checking")
        try:
            p.getBodyInfo(self.robot.id)
            p.getBodyInfo(body)
            p.performCollisionDetection()
            points = p.getClosestPoints(self.robot.id, body, distance=0.0)
            if points is None:
                points = p.getContactPoints(self.robot.id, body)
        except p.error as exc:
            raise CollisionQueryError(f"Collision query failed for body {body}: {exc}") from exc
        if points is None:
            raise CollisionQueryError(f"No collision data returned for body {body}")
        if not self._reported_query_fallback:
            print("[MotionPlanner] Recovered collision query using refreshed collision data.")
            self._reported_query_fallback = True
        return points

    def plan_motion(self, target_position, orientation_mode="vertical",
                    allowed_contact_body=None):
        """Sample a Cartesian line and validate the final tool orientation.

        Longer travel/retraction/approach moves are staged by CoordRobotController.
        A failed solve raises before any motor target is changed.
        """
        if orientation_mode not in ("vertical", "horizontal"):
            raise ValueError("orientation_mode must be 'vertical' or 'horizontal'")
        target = list(target_position)
        if len(target) != 3 or not all(math.isfinite(v) for v in target):
            raise MotionPlanningError("Target must contain three finite coordinates")
        start = self.robot.get_roller_position()
        previous = self.robot.get_joint_states()
        count = max(1, math.ceil(math.dist(start, target) / self.CARTESIAN_STEP))
        trajectory = []
        for i in range(1, count + 1):
            t = i / count
            point = [a + (b - a) * t for a, b in zip(start, target)]
            chosen = None
            for joints in self._candidate_configurations(point, previous, orientation_mode):
                if not self._valid_pose(joints, point, allowed_contact_body):
                    continue
                segment = self._interpolate(previous, joints)
                if all(not self._predicted_collision(q, allowed_contact_body) for q in segment):
                    chosen = joints, segment
                    break
            if chosen is None:
                raise MotionPlanningError(
                    f"No collision-free pose/path at {[round(v, 3) for v in point]}")
            previous, segment = chosen
            trajectory.extend(segment)
        return trajectory

    def plan_transit_motion(self, target_position):
        """Paint-off travel, with a checked reorientation if the straight path fails.

        Base yaw has real +/-pi stops. A Cartesian line can strand greedy IK at
        one stop even though the destination is reachable from the other side.
        Never wrap that joint or teleport it: try physical paths through home,
        then a bounded joint-space search if those are blocked. Plan the whole
        route before returning motor commands. Contact/vertical dip moves
        continue to use plan_motion without this detour.
        """
        if self.fixed_base:
            return self.plan_motion(target_position)  # no home-pose or RRT detours
        try:
            return self.plan_motion(target_position)
        except CollisionQueryError:
            raise
        except MotionPlanningError as direct_error:
            saved = self._save_joints()
            start = [position for position, velocity in saved]
            home = list(self.robot.home_joints)
            # The straight chassis path to home may cross the paint can.
            # Also try moving one base axis first, keeping the arm unchanged.
            via_x, via_y = list(start), list(start)
            via_x[0], via_y[1] = home[0], home[1]
            last_error = direct_error
            for stops in ([home], [via_x, home], [via_y, home]):
                trajectory = []
                try:
                    for stop in stops:
                        trajectory.extend(self.plan_to_joint_configuration(stop))
                        self._apply_joints(stop)
                    trajectory.extend(self.plan_motion(target_position))
                except CollisionQueryError:
                    raise
                except MotionPlanningError as detour_error:
                    last_error = detour_error
                    continue
                finally:
                    self._restore_joints(saved)
                print("[MotionPlanner] Transit rerouted through collision-checked home pose.")
                return trajectory
            try:
                trajectory = self._search_transit(target_position)
            except CollisionQueryError:
                raise
            except MotionPlanningError as search_error:
                raise MotionPlanningError(
                    f"Straight transit failed ({direct_error}); "
                    f"home detours failed ({last_error}); {search_error}") from search_error
            print("[MotionPlanner] Transit rerouted using collision-checked joint-space search.")
            return trajectory

    def plan_retraction_motion(self, target_position, allowed_contact_body=None):
        """Leave a wall even when a straight retreat traps the chassis at the can.

        Painting is off. Only the roller's existing active-wall contact is
        permitted during escape; the final pose must be completely clear.
        """
        try:
            return self.plan_motion(target_position, allowed_contact_body=allowed_contact_body)
        except CollisionQueryError:
            raise
        except MotionPlanningError:
            if self.fixed_base:
                raise  # no RRT escape - the base is not allowed to move
            trajectory = self._search_transit(target_position,
                                              allowed_contact_body=allowed_contact_body)
            print("[MotionPlanner] Wall retreat rerouted using collision-checked joint-space search.")
            return trajectory

    def _search_transit(self, target_position, max_iterations=1200, allowed_contact_body=None):
        """Bounded bidirectional RRT-Connect for paint-off, free-space travel.

        Search base position, yaw, shoulder pitch and extension. Wrist roll is
        zero and roller pitch cancels shoulder pitch, so the barrel stays
        upright. Yaw is a limited joint: distances never wrap across +/-pi.
        Every tree edge and every shortcut is checked with the same collision
        rules as ordinary motion; no simulation steps or motor commands occur.
        """
        start = self.robot.get_joint_states()
        if self._predicted_collision(start, allowed_contact_body):
            raise MotionPlanningError("Transit search starts in collision; cannot assume a safe escape")
        goals = [q for q in self._candidate_configurations(target_position, start)
                 if self._valid_pose(q, target_position, None)]
        if not goals:
            raise MotionPlanningError("Transit destination has no collision-free IK pose")

        scales = (0.5, 0.5, 1.0, 0.7, 0.35, 1.0, 0.7)
        def distance(a, b):
            return max(abs(x-y)/scale for x,y,scale in zip(a,b,scales))

        def edge_clear(a, b):
            return all(not self._predicted_collision(q, allowed_contact_body)
                       for q in self._interpolate(a, b))

        # Each goal IK solution is a separate root in the goal-side forest.
        forward = [(list(start), None)]
        backward = [(q, None) for q in goals]

        def extend(tree, target):
            nearest = min(range(len(tree)), key=lambda i: distance(tree[i][0], target))
            source = tree[nearest][0]
            gap = distance(source, target)
            if gap < 1e-9:
                return nearest, True
            fraction = min(1.0, 0.5 / gap)
            candidate = [a + fraction*(b-a) for a,b in zip(source,target)]
            if not edge_clear(source, candidate):
                return None, False
            tree.append((candidate, nearest))
            return len(tree)-1, fraction == 1.0

        def root_path(tree, index):
            path = []
            while index is not None:
                q, parent = tree[index]
                path.append(q)
                index = parent
            return path[::-1]

        rng = random.Random(5478)  # reproducible search; independent of caller RNG
        lower, upper = self.robot.lower_limits, self.robot.upper_limits
        for iteration in range(max_iterations):
            a, b = (forward, backward) if iteration % 2 == 0 else (backward, forward)
            if rng.random() < 0.2:
                sample = list(rng.choice(b)[0])
            else:
                sample = [rng.uniform(lo,hi) for lo,hi in zip(lower,upper)]
                sample[5] = 0.0
                sample[6] = -sample[3]
            index_a, _ = extend(a, sample)
            if index_a is None:
                continue
            meeting = a[index_a][0]
            for _ in range(64):
                index_b, connected = extend(b, meeting)
                if index_b is None:
                    break
                if not connected:
                    continue
                if a is forward:
                    path = root_path(a,index_a) + root_path(b,index_b)[-2::-1]
                else:
                    path = root_path(b,index_b) + root_path(a,index_a)[-2::-1]
                # Shortcut only when the entire shorter edge passes collision checks.
                for _ in range(60):
                    if len(path) <= 2:
                        break
                    i, j = sorted(rng.sample(range(len(path)), 2))
                    if j > i+1 and edge_clear(path[i],path[j]):
                        path[i+1:j] = []
                trajectory = []
                for first, second in zip(path,path[1:]):
                    trajectory.extend(self._interpolate(first,second))
                if not trajectory:
                    trajectory = [list(start)]
                # Revalidate the assembled path, including edges traversed in reverse.
                if (any(self._predicted_collision(q, allowed_contact_body) for q in trajectory) or
                        not self._valid_pose(trajectory[-1], target_position, None)):
                    raise MotionPlanningError("Transit search path failed final validation")
                return trajectory
        raise MotionPlanningError(f"No transit route found within {max_iterations} search iterations")

    def _candidate_configurations(self, target, reference, orientation_mode="vertical"):
        """Exact position IK for the supplied robot, with zero wrist roll.

        Positive shoulder pitch lowers the arm. Opposite roller pitch cancels
        that rotation, keeping the roller axis vertical at every waypoint.
        Search base locations, not unconverged/clamped numerical IK results.
        """
        x, y, z = target
        dz = z - self.SHOULDER_HEIGHT
        reach_min = self.FIXED_REACH + self.robot.lower_limits[4]
        reach_max = self.FIXED_REACH + self.robot.upper_limits[4]
        if abs(dz) > reach_max:
            return []
        horizontal_min = math.sqrt(max(0.0, reach_min ** 2 - dz ** 2))
        horizontal_max = math.sqrt(max(0.0, reach_max ** 2 - dz ** 2))
        bx0, by0 = reference[:2]
        xmin, xmax = self.robot.x_range
        ymin, ymax = self.robot.y_range

        def clamp_base(bx, by):
            return max(xmin, min(xmax, bx)), max(ymin, min(ymax, by))

        bases = [(bx0, by0), clamp_base(x, y)]
        direction = math.atan2(y - by0, x - bx0)
        angles = [reference[2], direction]
        angles += [i * math.pi / 8 for i in range(-8, 8)]

        if self.fixed_base:
            # The base stays where it is - only the arm solves the pose. A base driven onto a stop can
            # read a hair past it, which would make every candidate fail the joint limit check.
            bases = [clamp_base(bx0, by0)]
        else:
            # Search a wider neighborhood around the target instead of only the exact target
            # base or the current base. The original implementation was too narrow and could
            # reject valid wall cells that should be reachable from a nearby base pose.
            radius_steps = [0.0, 0.12, 0.24, 0.36, 0.48]
            for radius in radius_steps:
                for yaw in angles:
                    for sign_x, sign_y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        bases.append(clamp_base(x + sign_x * radius * math.cos(yaw),
                                                y + sign_y * radius * math.sin(yaw)))

            for horizontal in (horizontal_min + 0.02,
                               (horizontal_min + horizontal_max) / 2,
                               horizontal_max - 0.02):
                if horizontal < horizontal_min or horizontal > horizontal_max:
                    continue
                radius = horizontal + self.SHOULDER_OFFSET
                for yaw in angles:
                    bases.append(clamp_base(x - radius * math.cos(yaw),
                                            y - radius * math.sin(yaw)))

        candidates, seen = [], set()
        for bx, by in bases:
            key = (round(bx, 6), round(by, 6))
            if key in seen:
                continue
            seen.add(key)
            horizontal = math.hypot(x - bx, y - by) - self.SHOULDER_OFFSET
            if horizontal < 0:
                continue
            reach = math.hypot(horizontal, dz)
            extension = reach - self.FIXED_REACH
            pitch = math.atan2(-dz, horizontal)
            yaw = math.atan2(y - by, x - bx)
            # atan2 returns +pi on the negative X axis; this URDF uses 3.14159.
            yaw = max(self.robot.lower_limits[2], min(self.robot.upper_limits[2], yaw))
            roller_pitch = -pitch + (math.pi / 2 if orientation_mode == "horizontal" else 0)
            roller_pitch = (roller_pitch + math.pi) % (2 * math.pi) - math.pi
            joints = [bx, by, yaw, pitch, extension, 0.0, roller_pitch]
            if all(lo - 1e-9 <= q <= hi + 1e-9 for q, lo, hi in
                   zip(joints, self.robot.lower_limits, self.robot.upper_limits)):
                cost = sum(w * (a - b) ** 2 for w, a, b in
                           zip((2, 2, 1, 0.3, 0.5, 0.2, 0.3), joints, reference))
                candidates.append((cost, joints))
        return [q for _, q in sorted(candidates, key=lambda item: item[0])]

    def _save_joints(self):
        return [(s[0], s[1]) for s in p.getJointStates(self.robot.id, self.robot.joint_indices)]

    def _apply_joints(self, joints):
        for idx, position in zip(self.robot.joint_indices, joints):
            p.resetJointState(self.robot.id, idx, position)

    def _restore_joints(self, saved):
        for idx, (position, velocity) in zip(self.robot.joint_indices, saved):
            p.resetJointState(self.robot.id, idx, position, targetVelocity=velocity)
        p.performCollisionDetection()

    def _valid_pose(self, joints, target, allowed_contact_body):
        saved = self._save_joints()
        try:
            self._apply_joints(joints)
            # Check against the actual URDF, so incompatible geometry fails clearly.
            return (math.dist(self.robot.get_roller_position(), target) <= 0.001
                    and not self.current_collision(allowed_contact_body))
        finally:
            self._restore_joints(saved)

    def current_collision(self, allowed_contact_body=None, executing=False):
        """Only the roller may touch the active wall; always check it against the floor."""
        a, b = self.robot.get_roller_axis_segment()
        center_z = (a[2] + b[2]) / 2
        axis_z = max(-1.0, min(1.0, (a[2] - b[2]) / self.robot.ROLLER_LENGTH))
        # Lowest point of a cylinder, including radial extent when tilted.
        bottom = (center_z - self.robot.ROLLER_LENGTH / 2 * abs(axis_z)
                  - self.robot.ROLLER_RADIUS * math.sqrt(max(0.0, 1 - axis_z ** 2)))
        clearance = 0.003 if executing else self.FLOOR_CLEARANCE
        if bottom < self.floor_z + clearance - 1e-6:
            return True
        # One body id, or several (the active wall plus the wall across a reflex corner, whose face
        # is coplanar with it there - pressing the roller into one presses it into both).
        if allowed_contact_body is None:
            allowed = ()
        elif isinstance(allowed_contact_body, (tuple, list, set, frozenset)):
            allowed = allowed_contact_body
        else:
            allowed = (allowed_contact_body,)
        for body in self.collision_bodies:
            for point in self._collision_points(body):
                link, distance = point[3], point[8]
                if link == self.robot.roller_link_index:
                    tolerance = self.WALL_PENETRATION if body in allowed else 0.0005
                else:
                    tolerance = 0.002  # numerical tolerance for the chassis on the floor
                if distance < -tolerance:
                    return True
        return False

    def _predicted_collision(self, joints, allowed_contact_body=None):
        saved = self._save_joints()
        try:
            self._apply_joints(joints)
            return self.current_collision(allowed_contact_body)
        finally:
            self._restore_joints(saved)

    def plan_to_joint_configuration(self, target_joints):
        if len(target_joints) != len(self.robot.joint_indices) or not all(
                math.isfinite(q) and lo <= q <= hi for q, lo, hi in
                zip(target_joints, self.robot.lower_limits, self.robot.upper_limits)):
            raise MotionPlanningError("Invalid joint configuration")
        trajectory = self._interpolate(self.robot.get_joint_states(), target_joints)
        if any(self._predicted_collision(q) for q in trajectory):
            raise MotionPlanningError("Joint-space path collides; retract the roller first")
        return trajectory

    def _interpolate(self, start, end):
        count = max(1, math.ceil(max(abs(a - b) for a, b in zip(start, end)) / self.JOINT_STEP))
        return [[a + (b - a) * i / count for a, b in zip(start, end)]
                for i in range(1, count + 1)]
