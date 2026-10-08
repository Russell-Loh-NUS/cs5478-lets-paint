"""Entry point: runs the coverage/motion planning stack on top of the continuous coordinate-
based paint environment (simulation/paint_environment.py, simulation/paint_wall.py), instead of
the old grid-based room. All 5 coverage strategies and the motion planner are reused unchanged;
only the wall/mapping/execution layers are adapted (see mapping/coord_wall_map.py and
control/coord_robot_controller.py).
"""
import time

import pybullet as p

from simulation.paint_environment import PaintEnvironment, PaintConfig
from simulation.paint_can import PaintCan
from simulation.robot import Robot
from mapping.coord_wall_map import CoordWallMap
from mapping.coverage_map import CoverageMap
from planning.coverage_planner import CoveragePlanner, DipStep
from planning.motion_planner import MotionPlanner
from control.coord_robot_controller import CoordRobotController
from control.robot_controller import RobotController
from evaluation.coverage import coverage_percentage, coverage_error
from evaluation.path_length import PathLengthTracker

MAX_PAINT_ATTEMPTS = 3
MAX_REPLANS = 3
# Toggle between "nearest_neighbor", "lawnmower", "optimized", "astar", and "sweep" (see
# planning/coverage_planner.py) to compare coverage strategies. "sweep" paints directly along
# the roller's true footprint rather than the coarser 0.3m planning grid the others use, so it's
# the only one that doesn't structurally leave gaps between touches (single-pass: ~79% vs ~47%).
COVERAGE_STRATEGY = "bcd_astar"
WALL_ORDER = ["south", "east", "north", "west"]
# A* always runs its exact, exponential search now (no fallback) -- so when it's selected, use
# a coarser grid (close to the pre-roller-shrink cell size) to keep the per-wall cell count
# small enough to actually finish, instead of the much finer grid the other strategies use.
ASTAR_CELL_WIDTH = 1.0
ASTAR_CELL_HEIGHT = 0.5
# Keep the base's center at least this far from every wall so its 0.42x0.42 footprint actually
# clears the wall surface instead of being commanded to overlap it -- without this, position
# control stalls forever against real collision, burning the full per-waypoint step budget.
BASE_CLEARANCE = 0.27


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
            print(f"[main_lets_paint] Skipping cell ({step.row},{step.col}) after "
                  f"{MAX_PAINT_ATTEMPTS} failed attempts.")

        iteration += 1
        if iteration % 10 == 0:
            print(f"Progress: {coverage_percentage(wall_map):.1f}% "
                  f"({wall_map.painted_count()}/{total_cells} cells), "
                  f"path length so far: {path_tracker.get_total_length():.2f} m")
    return True


def _paint_wall(name, wall_map, env, robot, paint_can, motion_planner, path_tracker):
    """Plan and paint a single wall (replanning the remainder if a pass leaves gaps)."""
    wall = env.walls_by_name[name]
    env.focus_camera_on_wall(name)

    coverage_map = CoverageMap(wall_map)
    # wall_map doubles as the "wall" argument too: it exposes rows/cols/touch_point, exactly
    # what CoveragePlanner needs, on top of the continuous canvas underneath.
    coverage_planner = CoveragePlanner(coverage_map, wall_map)
    controller = CoordRobotController(robot, motion_planner, wall_map, path_tracker, env)

    print(f"\n=== Painting wall '{name}' ===")
    if not controller.ensure_dipped(paint_can):
        print(f"[{name}] Could not dip the roller; skipping this wall.")
        return

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

    print(f"[{name}] Coverage: {coverage_percentage(wall_map):.1f}%  Error: {coverage_error(wall_map):.1f}%")


def _room_bounds(env):
    """Bounding box of the room's walls in world x/y, for sizing the robot's reachable footprint."""
    xs, ys = [], []
    for w in env.walls:
        xs += [float(w.origin[0]), float(w.origin[0] + w.u_dir[0] * w.length)]
        ys += [float(w.origin[1]), float(w.origin[1] + w.u_dir[1] * w.length)]
    return (min(xs), max(xs)), (min(ys), max(ys))


def main():
    run_started_at = time.perf_counter()
    # patch_width scaled to our roller's actual diameter -- the paint environment's tiny
    # default (4mm) assumes a much smaller, more realistic roller and would never fill a
    # whole grid cell.
    env = PaintEnvironment(gui=True, paint=PaintConfig(patch_width=2 * Robot.ROLLER_RADIUS))

    x_range, y_range = _room_bounds(env)
    wall_height = max(w.height for w in env.walls)
    start_xy = ((x_range[0] + x_range[1]) / 2.0, (y_range[0] + y_range[1]) / 2.0)

    paint_can = PaintCan(position=(start_xy[0], start_xy[1], 0.0))
    # The base travels in an inset range (clear of the walls); the arm spans the remaining gap
    # out to the real wall surface, which touch_point() targets directly.
    base_x_range = (x_range[0] + BASE_CLEARANCE, x_range[1] - BASE_CLEARANCE)
    base_y_range = (y_range[0] + BASE_CLEARANCE, y_range[1] - BASE_CLEARANCE)
    robot = Robot(start_xy=start_xy, max_lift=wall_height, x_range=base_x_range, y_range=base_y_range)
    path_tracker = PathLengthTracker()
    motion_planner = MotionPlanner(robot, collision_bodies=[w.body_id for w in env.walls] + [env.floor_id])

    wall_maps = {}
    try:
        for name in WALL_ORDER:
            wall = env.walls_by_name[name]
            # Registered immediately (before execution) so a mid-wall connection loss still
            # leaves this wall's partial progress visible in the final summary below.
            # Cell size matches the roller's true footprint 1:1 (the original design intent:
            # one touch at a cell's center should fully cover it) -- more cells than a floored
            # grid, but no structural gaps for "optimized"/"lawnmower"/"nearest_neighbor" like a
            # coarser grid leaves. "astar" keeps its own coarser dedicated grid (see
            # ASTAR_CELL_WIDTH/HEIGHT above) since its exact search needs a small cell count.
            if COVERAGE_STRATEGY == "astar":
                cell_width, cell_height = ASTAR_CELL_WIDTH, ASTAR_CELL_HEIGHT
            else:
                cell_width, cell_height = 2 * Robot.ROLLER_RADIUS, Robot.ROLLER_LENGTH
            wall_maps[name] = CoordWallMap(wall, cell_width=cell_width, cell_height=cell_height,
                                            touch_spacing=2 * Robot.ROLLER_RADIUS,
                                            line_spacing=Robot.ROLLER_LENGTH)
            _paint_wall(name, wall_maps[name], env, robot, paint_can, motion_planner, path_tracker)
            if not p.isConnected():
                break
    except p.error as exc:
        # Most commonly the GUI window was closed mid-run, dropping the physics connection.
        print(f"Simulation connection lost ({exc}); stopping early with partial progress.")

    total_cells = sum(wm.total_cells() for wm in wall_maps.values())
    total_painted = sum(wm.painted_count() for wm in wall_maps.values())
    overall_coverage = 100.0 * total_painted / total_cells if total_cells else 0.0
    all_complete = all(wm.all_painted() for wm in wall_maps.values()) and len(wall_maps) == len(WALL_ORDER)

    print("\n=== Room painting summary ===")
    print("All walls complete!" if all_complete else "Room painting stopped before full coverage.")
    print(f"Overall coverage: {overall_coverage:.1f}% ({total_painted}/{total_cells} cells)")
    per_wall_area, total_area = env.painted_area()
    print(f"Ground-truth painted area (from canvas): {total_area:.3f} m^2 {per_wall_area}")
    print(f"Total path length travelled: {path_tracker.get_total_length():.2f} m")
    elapsed = time.perf_counter() - run_started_at
    print(f"Total runtime (environment init to painting finish): {elapsed:.2f} s")

    if not p.isConnected():
        return

    if all_complete:
        print("Moving robot back to the center of the room...")
        try:
            CoordRobotController(robot, motion_planner, None, path_tracker, env).return_to_center()
        except p.error as exc:
            print(f"Simulation connection lost while returning home ({exc}).")
            return

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        env.close()


if __name__ == "__main__":
    main()
