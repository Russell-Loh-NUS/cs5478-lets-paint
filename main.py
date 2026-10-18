"""Entry point: runs the coverage/motion planning stack on top of the continuous coordinate-
based paint environment (simulation/paint_environment.py, simulation/paint_wall.py), instead of
the old grid-based room. All 5 coverage strategies and the motion planner are reused unchanged;
only the wall/mapping/execution layers are adapted (see mapping/coord_wall_map.py and
control/coord_robot_controller.py).
"""
import sys
import time
from pathlib import Path

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
# Toggle between "nearest_neighbor", "lawnmower", "optimized", "astar", and "sweep", "grid_smooth", "prm", "optimized_prm", "bcd_astar"
COVERAGE_STRATEGY = "rl"
# Set to False to disable repaint/replanning after the first pass.
REPLANNING_ENABLED = False
WALL_ORDER = ["south", "east", "north", "west"]
# Optional CLI override for RL model path
RL_MODEL_PATH: str | None = None
# A* always runs its exact, exponential search now (no fallback) -- so when it's selected, use
# a coarser grid (close to the pre-roller-shrink cell size) to keep the per-wall cell count
# small enough to actually finish, instead of the much finer grid the other strategies use.
ASTAR_CELL_WIDTH = 1.0
ASTAR_CELL_HEIGHT = 0.5
# Chassis is 0.55 x 0.64 m; its half-diagonal is about 0.422 m. This also
# leaves a small margin when the base yaw rotates the chassis diagonally toward a wall.
BASE_CLEARANCE = 0.60


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if data:
                stream.write(data)
        for stream in self.streams:
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def configure_file_logging():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"program-{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"[main] Logging to {log_path}")
    return log_file


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
    # The roller radius is 0.06 m. Approach with only 1 mm of nominal penetration so the
    # barrel can register wall contact without driving 10 mm into the wall.
    coverage_planner = CoveragePlanner(
        coverage_map, wall_map, approach_offset=Robot.ROLLER_RADIUS - 0.001
    )
    controller = CoordRobotController(robot, motion_planner, wall_map, path_tracker, env)

    print(f"\n=== Painting wall '{name}' ===")
    if not controller.ensure_dipped(paint_can):
        print(f"[{name}] Could not dip the roller; skipping this wall.")
        return

    total_cells = wall_map.total_cells()
    attempt = 0
    # default RL model path (used when COVERAGE_STRATEGY == 'rl')
    rl_model_default = Path(__file__).resolve().parent / "models" / "ppo_coverage_final.zip"
    # allow CLI override
    model_path = RL_MODEL_PATH if RL_MODEL_PATH is not None else (str(rl_model_default) if rl_model_default.exists() else None)
    if REPLANNING_ENABLED:
        while not coverage_map.is_complete(threshold=0.8) and attempt < MAX_REPLANS:
            attempt += 1
            print(f"[{name}] Planning the full paint route (attempt {attempt}/{MAX_REPLANS}, "
                  f"{COVERAGE_STRATEGY}, dip-aware)...")
            # If using RL strategy, pass the trained model path into the planner
            if COVERAGE_STRATEGY == 'rl':
                if model_path is None:
                    print(f"[main] WARNING: RL model not found at {rl_model_default}; planner will error if required.")
                route = coverage_planner.plan_full_route(robot.get_roller_position(), paint_can,
                                                       RobotController.REDIP_DISTANCE, strategy=COVERAGE_STRATEGY,
                                                       rl_model_path=model_path)
            else:
                route = coverage_planner.plan_full_route(robot.get_roller_position(), paint_can,
                                                       RobotController.REDIP_DISTANCE, strategy=COVERAGE_STRATEGY)
            print(f"[{name}] Route planned: {len(route)} steps "
                  f"({sum(1 for s in route if isinstance(s, DipStep))} redips included).")
            coverage_planner.print_route(route)

            if not _execute_route(route, controller, paint_can, wall_map, path_tracker, total_cells):
                break

            if not coverage_map.is_complete(threshold=0.8):
                print(f"[{name}] Coverage at {coverage_percentage(wall_map):.1f}% after attempt {attempt}; "
                      "replanning the remaining under-painted cells (<80%)...")

        if not coverage_map.is_complete(threshold=0.8) and p.isConnected() and attempt >= MAX_REPLANS:
            print(f"[{name}] Still incomplete after {MAX_REPLANS} replanning attempts; giving up.")
    else:
        print(f"[{name}] Replanning disabled; running a single paint pass only.")
        if COVERAGE_STRATEGY == 'rl':
            if model_path is None:
                print(f"[main] WARNING: RL model not found at {rl_model_default}; planner will error if required.")
            route = coverage_planner.plan_full_route(robot.get_roller_position(), paint_can,
                                                   RobotController.REDIP_DISTANCE, strategy=COVERAGE_STRATEGY,
                                                   rl_model_path=model_path)
        else:
            route = coverage_planner.plan_full_route(robot.get_roller_position(), paint_can,
                                                   RobotController.REDIP_DISTANCE, strategy=COVERAGE_STRATEGY)
        print(f"[{name}] Route planned: {len(route)} steps "
              f"({sum(1 for s in route if isinstance(s, DipStep))} redips included).")
        coverage_planner.print_route(route)
        _execute_route(route, controller, paint_can, wall_map, path_tracker, total_cells)

    coverage = coverage_percentage(wall_map)
    print(f"[{name}] Coverage: {coverage:.1f}%  Error: {coverage_error(wall_map):.1f}%")
    if coverage < 100.0:
        uncovered, total = wall_map.uncovered_cells(limit=50)
        print(f"[{name}] Uncovered cells ({total} total): {uncovered[:10]}" + (" ..." if total > 10 else ""))
        if uncovered:
            for row, col, frac in uncovered[:20]:
                print(f"  - [{name}] cell ({row}, {col}) covered only {frac:.2%}")


def _room_bounds(env):
    """Bounding box of the room's walls in world x/y, for sizing the robot's reachable footprint."""
    xs, ys = [], []
    for w in env.walls:
        xs += [float(w.origin[0]), float(w.origin[0] + w.u_dir[0] * w.length)]
        ys += [float(w.origin[1]), float(w.origin[1] + w.u_dir[1] * w.length)]
    return (min(xs), max(xs)), (min(ys), max(ys))


def main(gui=True):
    log_file = configure_file_logging()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    run_started_at = time.perf_counter()
    try:
        # patch_width scaled to our roller's actual diameter -- the paint environment's tiny
        # default (4mm) assumes a much smaller, more realistic roller and would never fill a
        # whole grid cell.
        env = PaintEnvironment(gui=gui, paint=PaintConfig(patch_width=2 * Robot.ROLLER_RADIUS))
        try:
            _run(env, run_started_at)
        except KeyboardInterrupt:
            print("Simulation stopped by user.")
        finally:
            try:
                env.close()
            except p.error as exc:
                print(f"[cleanup] PyBullet disconnect failed: {exc}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def _run(env, run_started_at):

    x_range, y_range = _room_bounds(env)
    wall_height = max(w.height for w in env.walls)
    start_xy = ((x_range[0] + x_range[1]) / 2.0, (y_range[0] + y_range[1]) / 2.0)

    # Keep the refill can beside the compact chassis so the arm approaches it from the side.
    paint_can = PaintCan(position=(start_xy[0] - 0.8, start_xy[1], 0.0))
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
        # A failed state query does not by itself prove that the client disconnected.
        print(f"PyBullet call failed: {exc}; connected={p.isConnected()}")
        raise  # preserve the exact failing operation and full traceback

    total_cells = sum(wm.total_cells() for wm in wall_maps.values())
    total_painted = sum(wm.painted_count() for wm in wall_maps.values())
    overall_coverage = 100.0 * total_painted / total_cells if total_cells else 0.0
    all_complete = all(wm.all_painted() for wm in wall_maps.values()) and len(wall_maps) == len(WALL_ORDER)

    print("\n=== Room painting summary ===")
    print("All walls complete!" if all_complete else "Room painting stopped before full coverage.")
    print(f"Overall coverage: {overall_coverage:.1f}% ({total_painted}/{total_cells} cells)")
    for name, wm in wall_maps.items():
        coverage = coverage_percentage(wm)
        if coverage < 100.0:
            print(f"[{name}] incomplete wall: coverage={coverage:.1f}%")
            uncovered, total = wm.uncovered_cells(limit=20, threshold=0.8)
            if uncovered:
                print(f"[{name}] cells below 80% coverage: {[(r, c, f'{frac:.2%}') for r, c, frac in uncovered]}")
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
            print(f"PyBullet return-home call failed: {exc}; connected={p.isConnected()}")
            raise

    if not env.gui:
        return  # headless full-run diagnostics must finish instead of waiting for a window
    while True:
        env.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the wall-painting simulation")
    parser.add_argument("--headless", action="store_true", help="Run full coverage without the GUI")
    parser.add_argument("--disable-replanning", action="store_true",
                        help="Disable repaint/replanning after the first pass")
    parser.add_argument("--strategy", type=str, help="Coverage strategy to use (e.g. rl, lawnmower)")
    parser.add_argument("--rl-model", type=str, help="Path to RL model to use when strategy=rl")
    args = parser.parse_args()
    if args.disable_replanning:
        REPLANNING_ENABLED = False
    if args.strategy:
        COVERAGE_STRATEGY = args.strategy
    # If provided, let the rl model path override the default used by the planner.
    if args.rl_model:
        RL_MODEL_PATH = args.rl_model
    main(gui=not args.headless)
