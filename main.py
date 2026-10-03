"""Entry point: builds the environment/robot, then plans and paints the wall end to end."""
import time

import pybullet as p

from simulation.environment import Environment
from simulation.wall import Wall
from simulation.paint_can import PaintCan
from simulation.robot import Robot
from perception.wall_detection import WallDetector
from mapping.wall_map import WallMap
from mapping.coverage_map import CoverageMap
from planning.coverage_planner import CoveragePlanner, DipStep
from planning.motion_planner import MotionPlanner
from control.robot_controller import RobotController
from evaluation.coverage import coverage_percentage, coverage_error
from evaluation.path_length import PathLengthTracker

MAX_PAINT_ATTEMPTS = 3
MAX_REPLANS = 3
# Toggle between "nearest_neighbor", "lawnmower", "optimized" (2-opt tour + DP-optimal dip
# placement), and "astar" (provably optimal visiting order, exact but only practical for
# small cell counts -- see CoveragePlanner.ASTAR_MAX_CELLS) to compare coverage strategies.
COVERAGE_STRATEGY = "optimized"
OPPOSITE_WALL = {"back": "front", "front": "back", "left": "right", "right": "left"}
# Paint order chosen so each move is to an adjacent (90 deg away) wall, not straight across.
WALL_ORDER = ["back", "right", "front", "left"]


def _execute_route(route, controller, paint_can, wall_map, path_tracker, total_cells):
    """Run one planned route. Returns False if the simulation connection was lost."""
    iteration = 0
    for step in route:
        if not p.isConnected():
            print("Simulation window was closed; stopping early.")
            return False

        # Safety net: the planned route estimates distance with straight-line hops, so
        # actual joint-space travel can drift; redip early if we're really out of paint.
        if controller.needs_redip():
            print(f"Roller has travelled {path_tracker.distance_since_dip():.2f}m since last dip; redipping...")
            if not controller.ensure_dipped(paint_can):
                print("Could not redip the roller; stopping before painting without paint.")
                return True

        if isinstance(step, DipStep):
            if not controller.ensure_dipped(paint_can):
                print("Could not redip the roller; stopping before painting without paint.")
                return True
            continue

        success = False
        for _ in range(MAX_PAINT_ATTEMPTS):
            success = controller.paint_target(step)
            if success:
                break
        if not success:
            print(f"[main] Skipping cell ({step.row},{step.col}) after {MAX_PAINT_ATTEMPTS} failed attempts.")

        iteration += 1
        if iteration % 10 == 0:
            print(f"Progress: {coverage_percentage(wall_map):.1f}% "
                    f"({wall_map.painted_count()}/{total_cells} cells), "
                    f"path length so far: {path_tracker.get_total_length():.2f} m")
    return True


def _paint_wall(name, walls, robot, paint_can, motion_planner, path_tracker, env):
    """Plan and paint a single wall (replanning the remainder if a pass leaves gaps)."""
    wall = walls[name]
    for other_name, other_wall in walls.items():
        other_wall.set_visible(other_name != OPPOSITE_WALL[name])
    env.focus_camera_on_wall(wall)

    wall_map = WallMap(wall)
    coverage_map = CoverageMap(wall_map)
    wall_detector = WallDetector(wall, robot, threshold=0.05)
    coverage_planner = CoveragePlanner(coverage_map, wall)
    controller = RobotController(robot, motion_planner, wall_detector, wall_map, path_tracker)

    print(f"\n=== Painting wall '{name}' ===")
    if not controller.ensure_dipped(paint_can):
        print(f"[{name}] Could not dip the roller; skipping this wall.")
        return wall_map

    total_cells = wall_map.total_cells()
    attempt = 0
    while not coverage_map.is_complete() and attempt < MAX_REPLANS:
        attempt += 1
        print(f"[{name}] Planning the full paint route (attempt {attempt}/{MAX_REPLANS}, "
                f"{COVERAGE_STRATEGY}, dip-aware)...")
        route = coverage_planner.plan_full_route(robot.get_roller_position(), paint_can,
                                                RobotController.REDIP_DISTANCE, strategy=COVERAGE_STRATEGY)
        print(f"[{name}] Route planned: {len(route)} steps "
                f"({sum(1 for s in route if isinstance(s, DipStep))} redips included).")
        coverage_planner.print_route(route)

        if not _execute_route(route, controller, paint_can, wall_map, path_tracker, total_cells):
            break

        if not coverage_map.is_complete():
            print(f"[{name}] Coverage at {coverage_percentage(wall_map):.1f}% after attempt {attempt}; "
                    "replanning the remaining unpainted cells...")

    if not coverage_map.is_complete() and p.isConnected() and attempt >= MAX_REPLANS:
        print(f"[{name}] Still incomplete after {MAX_REPLANS} replanning attempts; giving up.")

    print(f"[{name}] Coverage: {coverage_percentage(wall_map):.1f}% Error: {coverage_error(wall_map):.1f}%")
    return wall_map


def main():
    run_started_at = time.perf_counter()
    env = Environment(gui=True)
    room_half = env.floor_size / 2.0
    # All 4 walls of the room (back/front run along x; left/right run along y). The robot's
    # yaw joint lets the same arm reorient to face any of them, so every wall gets painted in
    # turn; visibility shows the current wall + its two side walls, hiding the opposite one.
    wall_kwargs = dict(length=10.0, height=3.0, grid_width=2 * Robot.ROLLER_RADIUS,
                        grid_height=Robot.ROLLER_LENGTH)
    walls = {
        "back": Wall(axis="y", position=room_half, name="back", **wall_kwargs),
        "front": Wall(axis="y", position=-room_half, name="front", **wall_kwargs),
        "left": Wall(axis="x", position=-room_half, name="left", **wall_kwargs),
        "right": Wall(axis="x", position=room_half, name="right", **wall_kwargs),
    }

    # Demo obstacle (e.g. a window): a non-paintable 2x2 block at the center of the back wall.
    back_wall = walls["back"]
    mid_row, mid_col = back_wall.rows // 2, back_wall.cols // 2
    for row in (mid_row - 1, mid_row):
        for col in (mid_col - 1, mid_col):
            back_wall.mark_obstacle(row, col)
    print(f"Obstacle placed at {sorted(back_wall.obstacle_cells)} (not paintable).")

    for name, w in walls.items():
        w.set_visible(name != OPPOSITE_WALL[WALL_ORDER[0]])
    env.finish_setup()

    paint_can = PaintCan(position=(1.5, 1.5, 0.0))
    # Lift range sized for the tallest wall; the yaw joint handles facing each wall in turn.
    robot = Robot(start_xy=(0.0, 0.0), max_lift=max(w.height for w in walls.values()))
    path_tracker = PathLengthTracker()
    motion_planner = MotionPlanner(robot, collision_bodies=[w.body_id for w in walls.values()] + [env.floor_id])

    wall_maps = {}
    try:
        for name in WALL_ORDER:
            wall_maps[name] = _paint_wall(name, walls, robot, paint_can, motion_planner, path_tracker, env)
            if not p.isConnected():
                break
    except p.error as exc:
        # Most commonly the GUI window was closed mid-run, dropping the physics connection.
        print(f"Simulation connection lost ({exc}); stopping early with partial progress.")

    total_cells = sum(wm.total_cells() for wm in wall_maps.values())
    total_painted = sum(wm.painted_count() for wm in wall_maps.values())
    total_violations = sum(wm.painted_obstacle_count() for wm in wall_maps.values())
    overall_coverage = 100.0 * total_painted / total_cells if total_cells else 0.0
    all_complete = all(wm.all_painted() for wm in wall_maps.values()) and len(wall_maps) == len(WALL_ORDER)

    print("\n=== Room painting summary ===")
    print("All walls complete!" if all_complete else "Room painting stopped before full coverage.")
    print(f"Overall coverage: {overall_coverage:.1f}% ({total_painted}/{total_cells} cells)")
    if total_violations > 0:
        print(f"Obstacle violations: {total_violations} obstacle cell(s) painted by mistake!")
    print(f"Total path length travelled: {path_tracker.get_total_length():.2f} m")
    elapsed = time.perf_counter() - run_started_at
    print(f"Total runtime (environment init to painting finish): {elapsed:.2f} s")

    if not p.isConnected():
        return

    if all_complete:
        print("Moving robot back to the center of the room...")
        try:
            RobotController(robot, motion_planner, None, None, path_tracker).return_to_center()
        except p.error as exc:
            print(f"Simulation connection lost while returning home ({exc}).")
            return

    try:
        while True:
            env.step()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        p.disconnect()


if __name__ == "__main__":
    main()