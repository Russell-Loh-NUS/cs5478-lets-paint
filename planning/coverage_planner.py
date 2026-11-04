"""Coverage planner for the active strategies used in this project.

The current planner supports the following route builders:
- "sweep": continuous boustrophedon coverage along the wall's real (u, v) surface.
- "optimized_prm": PRM-style continuous coverage with optimal dip placement.
- "bcd_astar": region-aware boustrophedon coverage with A* transitions between regions.
- "rl": RL-guided route planning via a pretrained policy model.

All strategies respect the same hard dip-distance budget so their results are directly
comparable in path length and redip count.
"""
import heapq
import math
from typing import Optional


class CoverageTarget:
    def __init__(self, row, col, position, orientation):
        self.row = row
        self.col = col
        self.position = position
        self.orientation = orientation


class DipStep:
    """A marker in the planned route: redip before continuing to the next target."""


class CoveragePlanner:
    STRATEGIES = ("sweep", "optimized_prm", "bcd_astar", "rl")
    # How many nearest neighbors each PRM sample connects to in the roadmap graph.
    PRM_K_NEIGHBORS = 6

    def __init__(self, coverage_map, wall, approach_offset=0.05):
        self.coverage_map = coverage_map
        self.wall = wall
        self.approach_offset = approach_offset

    def plan_full_route(self, start_position, paint_can, redip_distance, strategy="sweep", rl_model_path: Optional[str] = None, rl_max_steps: int = 2000):
        """Read the whole unpainted set and return a dip-aware route, using the given strategy."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown coverage strategy '{strategy}', expected one of {self.STRATEGIES}")

        if strategy == "sweep":
            return self._build_route_sweep(start_position, paint_can, redip_distance)
        if strategy == "optimized_prm":
            return self._build_route_optimized_prm(start_position, paint_can, redip_distance)
        if strategy == "bcd_astar":
            return self._build_route_bcd_astar(start_position, paint_can, redip_distance)
        if strategy == "rl":
            return self._build_route_rl(start_position, paint_can, redip_distance, rl_model_path=rl_model_path, rl_max_steps=rl_max_steps)
        raise ValueError(f"Unhandled coverage strategy '{strategy}'")

    def print_route(self, route):
        """Print a human-readable log of the planned route, for inspecting how it was planned."""
        print(f"[CoveragePlanner] Planned route ({len(route)} steps):")
        for i, step in enumerate(route, start=1):
            if isinstance(step, DipStep):
                print(f"  {i:4d}. DIP at paint can")
            else:
                x, y, z = step.position
                print(f"  {i:4d}. PAINT cell (row={step.row}, col={step.col}) "
                      f"-> ({x:.2f}, {y:.2f}, {z:.2f}), orientation={step.orientation}")

    def _nearest_order(self, remaining, cell_positions, start_position):
        """Greedy nearest-cell visiting order, ignoring dip stops (seed tour for 2-opt)."""
        remaining = list(remaining)
        order = []
        current_position = list(start_position)
        while remaining:
            best_cell, best_distance = None, None
            for cell in remaining:
                distance = math.dist(current_position, cell_positions[cell])
                if best_distance is None or distance < best_distance:
                    best_cell, best_distance = cell, distance
            order.append(best_cell)
            remaining.remove(best_cell)
            current_position = cell_positions[best_cell]
        return order

    def _two_opt_improve(self, order, cell_positions, start_position, max_iterations=200):
        """Local-search tour improvement over the whole route: repeatedly reverse whichever
        segment shortens the open path (start -> cells...). Not provably optimal, but closes
        most of the gap to the true shortest path versus a raw nearest-neighbor tour."""
        points = [list(start_position)] + [cell_positions[c] for c in order]
        n = len(points)
        perm = list(range(n))

        def dist(a, b):
            return math.dist(points[perm[a]], points[perm[b]])

        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    removed = dist(i - 1, i)
                    added = dist(i - 1, j)
                    if j + 1 < n:
                        removed += dist(j, j + 1)
                        added += dist(i, j + 1)
                    if added < removed - 1e-9:
                        perm[i:j + 1] = perm[i:j + 1][::-1]
                        improved = True

        return [order[k - 1] for k in perm[1:]]


    def _insert_optimal_dips(self, ordered_cells, cell_positions, start_position, paint_can, redip_distance):
        """Given a fixed visiting order, choose where (if anywhere) to dip so total added
        travel is minimized, subject to never exceeding redip_distance since the last dip.
        Dipping is free to happen at any point, not just when forced -- solved exactly with a
        shortest-path DP over candidate dip points."""
        positions = [list(start_position)] + [cell_positions[c] for c in ordered_cells]
        can_position = paint_can.get_dip_position()
        n = len(ordered_cells)

        d = [0.0] * (n + 1)
        for i in range(1, n + 1):
            d[i] = math.dist(positions[i - 1], positions[i])
        prefix = [0.0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + d[i]

        # extra[j]: added distance from dipping right after cell j, before continuing to j+1.
        extra = [0.0] * n
        for j in range(1, n):
            direct = d[j + 1]
            via_can = math.dist(positions[j], can_position) + math.dist(can_position, positions[j + 1])
            extra[j] = via_can - direct

        def stretch(j, i):
            """Real travel distance from the reset point after breakpoint j through cell i."""
            if j == 0:
                return prefix[i]
            return math.dist(can_position, positions[j + 1]) + (prefix[i] - prefix[j + 1])

        infinity = float("inf")
        best_cost = [infinity] * (n + 1)
        best_cost[0] = 0.0
        came_from = [None] * (n + 1)
        for i in range(1, n + 1):
            for j in range(i):
                if stretch(j, i) > redip_distance + 1e-9:
                    continue
                cost = best_cost[j] + (extra[j] if j > 0 else 0.0)
                if cost < best_cost[i]:
                    best_cost[i] = cost
                    came_from[i] = j

        dip_after = set()
        if best_cost[n] < infinity:
            j = came_from[n]
            while j is not None and j > 0:
                dip_after.add(j)
                j = came_from[j]
        else:
            # Shouldn't normally happen (would mean even a single fresh-paint hop is too long);
            # fall back to a simple "dip only when forced" rule instead of failing outright.
            print("[CoveragePlanner] Warning: optimal dip placement infeasible; falling back to greedy redip.")
            distance_since_dip = 0.0
            for j in range(1, n):
                distance_since_dip += d[j]
                if distance_since_dip + d[j + 1] > redip_distance:
                    dip_after.add(j)
                    distance_since_dip = 0.0

        route = []
        for idx, cell in enumerate(ordered_cells, start=1):
            row, col = cell
            route.append(CoverageTarget(row, col, positions[idx], orientation="vertical"))
            if idx in dip_after:
                route.append(DipStep())
        return route

    def _build_route_fixed_order(self, ordered_cells, cell_positions, start_position, paint_can, redip_distance):
        route = []
        current_position = list(start_position)
        distance_since_dip = 0.0
        dip_position = paint_can.get_dip_position()

        for cell in ordered_cells:
            position = cell_positions[cell]
            distance = math.dist(current_position, position)

            if distance_since_dip + distance > redip_distance:
                route.append(DipStep())
                current_position = dip_position
                distance_since_dip = 0.0
                distance = math.dist(current_position, position)

            row, col = cell
            route.append(CoverageTarget(row, col, position, orientation="vertical"))
            distance_since_dip += distance
            current_position = position

        return route


    def _filter_unpainted_targets(self, points):
        """Keep the sweep-style strategies deterministic and wall-aligned by removing
        already-painted cells and any repeated samples from the same grid cell."""
        seen = set()
        filtered = []
        for row, col, position in points:
            if self.wall.is_painted(row, col):
                continue
            key = (row, col)
            if key in seen:
                continue
            seen.add(key)
            filtered.append((row, col, position))
        return filtered

    def _normalize_sweep_targets(self, points):
        """Make the sweep follow the same wall-cell target convention as the motion-feasible
        strategies: keep the ordered sweep, but pin each target to the cell's touch point so the
        robot never attempts raw continuous edge samples that the IK solver rejects."""
        normalized = []
        seen = set()
        for row, col, _ in points:
            if self.wall.is_painted(row, col):
                continue
            key = (row, col)
            if key in seen:
                continue
            seen.add(key)
            normalized.append((row, col, self.wall.touch_point(row, col, offset=self.approach_offset)))
        return normalized

    def _build_route_sweep(self, start_position, paint_can, redip_distance):
        """Continuous boustrophedon sweep directly over the wall's real surface (see
        CoordWallMap.continuous_targets) -- the visiting order IS the sweep, so there's no
        TSP ordering step here, just dip insertion along the fixed sweep order."""
        if not hasattr(self.wall, "continuous_targets"):
            raise ValueError("The 'sweep' strategy requires a continuous wall map (e.g. "
                              "CoordWallMap) that exposes continuous_targets().")
        points = self._normalize_sweep_targets(self.wall.continuous_targets())
        return self._build_route_fixed_points(points, start_position, paint_can, redip_distance)

    def _build_route_fixed_points(self, points, start_position, paint_can, redip_distance):
        """Like _build_route_fixed_order, but for a plain ordered list of (row, col, position)
        samples -- continuous sweep waypoints can repeat a (row, col) grid cell, so a dict
        keyed by cell (as the other strategies use) can't represent them."""
        route = []
        current_position = list(start_position)
        distance_since_dip = 0.0
        dip_position = paint_can.get_dip_position()

        for row, col, position in points:
            distance = math.dist(current_position, position)

            if distance_since_dip + distance > redip_distance:
                route.append(DipStep())
                current_position = dip_position
                distance_since_dip = 0.0
                distance = math.dist(current_position, position)

            route.append(CoverageTarget(row, col, position, orientation="vertical"))
            distance_since_dip += distance
            current_position = position

        return route

    def _build_route_rl(self, start_position, paint_can, redip_distance, rl_model_path: Optional[str] = None, rl_max_steps: int = 200):
        """Placeholder for RL-driven coverage planning.

        The RL planner is intentionally a runtime-driven strategy: an external RL loop
        using `planning.rl_env.RLCoverageEnv` should be used to choose actions. This
        placeholder returns an empty route so the classic planner flow can handle
        calls gracefully; the RL integration lives in `planning/rl_env.py`.
        """
        # RL-driven planning: use a trained SB3 policy that returns a continuous
        # action [x, y, dip_score] where x,y are world-coordinates for the next
        # target point and dip_score in [0,1] indicates whether to dip (>0.5).
        # The planner will decode the action into the nearest wall grid cell and
        # build a dip-aware route until the coverage target is reached or
        # `rl_max_steps` is exhausted.
        import os
        # Safety: ensure OpenMP duplicate-lib is allowed and limit threads to avoid
        # runtime conflicts on macOS (common with multiple OpenMP-linked libs).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            print(f"[CoveragePlanner] Warning: failed to import stable_baselines3: {e}")
            return []
        import numpy as _np

        def build_obs(covered_set):
            # Approximate observation expected by the RL policy: coverage fraction
            # (simulated by covered_set / total_cells), paint_remaining (use redip_distance
            # as a conservative proxy), and paintable_area (in m^2).
            # CoverageMap doesn't expose total_cells; use the wall map adapter instead.
            total_cells = self.wall.total_cells()
            covered = len(covered_set)
            coverage_fraction = float(covered) / float(total_cells) if total_cells > 0 else 0.0
            # paint_remaining not tracked here; use redip_distance as a proxy for remaining
            paint_remaining = float(redip_distance)
            # paintable area approximate: number of cells (not physical area)
            paintable_area = float(total_cells)
            return {
                "coverage": _np.array([coverage_fraction], dtype=_np.float32),
                "paint_remaining": _np.array([paint_remaining], dtype=_np.float32),
                "paintable_area": _np.array([paintable_area], dtype=_np.float32),
            }

        if rl_model_path is None:
            raise ValueError("rl_model_path must be provided for 'rl' strategy")

        # Force CPU device to avoid CUDA library issues on machines without GPU
        try:
            model = PPO.load(rl_model_path, env=None, device="cpu")
        except Exception as e:
            print(f"[CoveragePlanner] Warning: failed to load RL model '{rl_model_path}': {e}")
            return []
        print(f"[CoveragePlanner][RL] Loaded model from: {rl_model_path}")
        actions_taken = []
        route = []
        painted = set()
        total_cells = self.wall.total_cells()
        max_steps = rl_max_steps
        obs = build_obs(painted)
        last_position = list(start_position)

        for step in range(max_steps):
            remaining = [cell for cell in self.coverage_map.unpainted_cells(threshold=0.8) if cell not in painted]
            if not remaining:
                print(f"[CoveragePlanner][RL] No remaining uncovered cells after {step} RL steps; stopping planner.")
                break

            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception as e:
                print(f"[CoveragePlanner] Warning: RL model prediction failed at step {step}: {e}")
                break
            actions_taken.append(action)
            a = _np.asarray(action).flatten()
            if a.size < 3:
                break
            x, y, dip_score = float(a[0]), float(a[1]), float(a[2])
            dip = bool(dip_score > 0.5)

            # Prefer the next target to be near the current wall progress, not an arbitrary jump
            # across the entire wall. This keeps the RL planner behaving like a wall sweep while
            # still using the model output as a guide.
            best = None
            best_key = None
            for r, c in remaining:
                pos = self.wall.touch_point(r, c, offset=self.approach_offset)
                action_distance = math.dist((x, y), (pos[0], pos[1]))
                continuity_distance = math.dist(last_position, pos)
                # A low continuity distance is preferred, but the model still influences the choice.
                key = (continuity_distance + 0.25 * action_distance, action_distance)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (r, c)

            if best is None:
                break
            row, col = best
            if (row, col) in painted:
                print(f"[CoveragePlanner][RL] Duplicate target selected; retrying with nearest unpainted cell.")
                row, col = min(remaining, key=lambda cell: math.dist(last_position, self.wall.touch_point(cell[0], cell[1], offset=self.approach_offset)))

            position = self.wall.touch_point(row, col, offset=self.approach_offset)
            # The controller currently validates paint success only for near-vertical roller
            # contact against the wall. Use the same orientation for RL-generated paint targets;
            # travel/retraction remain vertical, so the motion planner can actually solve the IK.
            route.append(CoverageTarget(row, col, position, orientation="vertical"))
            painted.add((row, col))
            last_position = list(position)
            if dip:
                route.append(DipStep())

            obs = build_obs(painted)

        if not route:
            print("[CoveragePlanner][RL] Generated no route; this is a planner issue, not a robot reachability issue.")

        # Log the raw actions and their mapped cells for auditability
        try:
            print(f"[CoveragePlanner][RL] Actions taken ({len(actions_taken)}):")
            for i, a in enumerate(actions_taken, start=1):
                arr = _np.asarray(a).flatten()
                if arr.size >= 3:
                    print(f"  {i:3d}. action=[{arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f}]")
                else:
                    print(f"  {i:3d}. action={arr}")
        except Exception:
            pass

        print(f"[CoveragePlanner][RL] Mapped route: {len(route)} steps, {sum(1 for s in route if isinstance(s, DipStep))} redips")
        return route

    def _chaikin_smooth(self, points, iterations=1, ratio=0.25):
        """Chaikin's corner-cutting: each edge is replaced by two points nearer its endpoints,
        turning a jagged polyline into a smooth curve. Each new point inherits the (row, col)
        of whichever original endpoint it's nearer to, for the shared completion-tracking API.
        One iteration roughly doubles the point count, so it's kept low by default."""
        pts = points
        for _ in range(iterations):
            if len(pts) < 3:
                break
            smoothed = [pts[0]]
            for (row_a, col_a, a), (row_b, col_b, b) in zip(pts, pts[1:]):
                q = [a[i] + (b[i] - a[i]) * ratio for i in range(3)]
                r = [a[i] + (b[i] - a[i]) * (1.0 - ratio) for i in range(3)]
                smoothed.append((row_a, col_a, q))
                smoothed.append((row_b, col_b, r))
            smoothed.append(pts[-1])
            pts = smoothed
        return pts

    def _interpolate_roadmap_path(self, graph, positions, samples, order):
        """Return a dense, continuous list of wall waypoints along the roadmap path, not just a
        sparse set of sampled nodes. Each graph edge is interpolated into a short chain of points
        so the robot moves continuously instead of jumping between isolated contact points."""
        ordered_points = []
        current_node = None
        for node in order:
            hop_nodes = [node] if current_node is None else self._roadmap_astar(graph, positions, current_node, node)
            for a_idx, b_idx in zip(hop_nodes, hop_nodes[1:]):
                a = positions[a_idx]
                b = positions[b_idx]
                distance = math.dist(a, b)
                step_count = max(1, int(math.ceil(distance / max(self.wall.touch_spacing, 0.05))))
                for step in range(1, step_count + 1):
                    t = step / step_count
                    point = [a[i] + (b[i] - a[i]) * t for i in range(3)]
                    row, col = self.wall.world_to_grid(point)
                    if self.wall.is_painted(row, col):
                        continue
                    ordered_points.append((row, col, [point[0], point[1], point[2]]))
            current_node = node
        return ordered_points

    def _build_route_prm(self, start_position, paint_can, redip_distance):
        """Continuous A*, option 2 (sampling-based): sample valid wall points, connect them into a
        roadmap, then interpolate the graph edges so the robot moves continuously along the path
        instead of only visiting isolated sample centers."""
        samples, positions, graph, order = self._build_prm_roadmap(start_position)
        if not samples:
            return []

        ordered_points = self._interpolate_roadmap_path(graph, positions, samples, order)
        if not ordered_points:
            return []
        return self._insert_optimal_dips_points(ordered_points, start_position, paint_can, redip_distance)

    def _build_route_optimized_prm(self, start_position, paint_can, redip_distance):
        """"optimized" PRM coverage with continuous motion between roadmap samples.

        We keep the PRM path construction but normalize the geometric waypoints to the wall's
        touch-point convention so each interpolated point remains physically reachable.
        """
        samples, positions, graph, order = self._build_prm_roadmap(start_position)
        if not samples:
            return []

        ordered_points = self._interpolate_roadmap_path(graph, positions, samples, order)
        if not ordered_points:
            return []

        normalized_points = []
        seen = set()
        for row, col, position in ordered_points:
            key = (row, col)
            if key in seen:
                continue
            seen.add(key)
            normalized_points.append((row, col, self.wall.touch_point(row, col, offset=self.approach_offset)))

        return self._insert_optimal_dips_points(normalized_points, start_position, paint_can, redip_distance)

    def _build_prm_roadmap(self, start_position):
        """Shared by "prm" and "optimized_prm": sample the wall, connect each sample to its K
        nearest neighbors into a roadmap graph, and pick a visiting order over the roadmap
        nodes (nearest-neighbor + 2-opt, same heuristic "optimized" uses -- generic over any
        hashable "cell" key, including plain indices). Returns (samples, positions, graph,
        order); samples/positions may be empty if there's nothing left to paint."""
        samples = self.wall.prm_samples() if hasattr(self.wall, "prm_samples") else None
        if samples is None:
            raise ValueError("This strategy requires a continuous wall map (e.g. CoordWallMap) "
                              "that exposes prm_samples().")
        if not samples:
            return samples, [], [], []
        positions = [s[2] for s in samples]
        n = len(positions)

        # Roadmap: connect each sample to its K nearest neighbors (classic PRM "connect the
        # dots"), edges weighted by straight-line distance.
        graph = [[] for _ in range(n)]
        for i in range(n):
            neighbors = sorted(range(n), key=lambda j: math.dist(positions[i], positions[j]))[1:self.PRM_K_NEIGHBORS + 1]
            for j in neighbors:
                d = math.dist(positions[i], positions[j])
                graph[i].append((j, d))
                graph[j].append((i, d))

        index_positions = {i: positions[i] for i in range(n)}
        order = self._nearest_order(list(range(n)), index_positions, start_position)
        order = self._two_opt_improve(order, index_positions, start_position)
        return samples, positions, graph, order

    def _insert_optimal_dips_points(self, ordered_points, start_position, paint_can, redip_distance):
        """Like _insert_optimal_dips, but for a plain ordered list of (row, col, position)
        points -- continuous/roadmap waypoints can repeat a (row, col) cell, so they can't be
        used as a dict key the way the grid strategies' (row, col) cells are."""
        positions = [list(start_position)] + [p[2] for p in ordered_points]
        can_position = paint_can.get_dip_position()
        n = len(ordered_points)

        d = [0.0] * (n + 1)
        for i in range(1, n + 1):
            d[i] = math.dist(positions[i - 1], positions[i])
        prefix = [0.0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + d[i]

        extra = [0.0] * n
        for j in range(1, n):
            direct = d[j + 1]
            via_can = math.dist(positions[j], can_position) + math.dist(can_position, positions[j + 1])
            extra[j] = via_can - direct

        def stretch(j, i):
            if j == 0:
                return prefix[i]
            return math.dist(can_position, positions[j + 1]) + (prefix[i] - prefix[j + 1])

        infinity = float("inf")
        best_cost = [infinity] * (n + 1)
        best_cost[0] = 0.0
        came_from = [None] * (n + 1)
        for i in range(1, n + 1):
            for j in range(i):
                if stretch(j, i) > redip_distance + 1e-9:
                    continue
                cost = best_cost[j] + (extra[j] if j > 0 else 0.0)
                if cost < best_cost[i]:
                    best_cost[i] = cost
                    came_from[i] = j

        dip_after = set()
        if best_cost[n] < infinity:
            j = came_from[n]
            while j is not None and j > 0:
                dip_after.add(j)
                j = came_from[j]
        else:
            print("[CoveragePlanner] Warning: optimal dip placement infeasible; falling back to greedy redip.")
            distance_since_dip = 0.0
            for j in range(1, n):
                distance_since_dip += d[j]
                if distance_since_dip + d[j + 1] > redip_distance:
                    dip_after.add(j)
                    distance_since_dip = 0.0

        route = []
        for idx, (row, col, position) in enumerate(ordered_points, start=1):
            route.append(CoverageTarget(row, col, position, orientation="vertical"))
            if idx in dip_after:
                route.append(DipStep())
        return route

    def _roadmap_astar(self, graph, positions, start, goal):
        """Generic A* shortest path over a sparse graph (the PRM roadmap) -- needed because the
        roadmap only connects nearby samples, so distant targets may require several hops."""
        def heuristic(i):
            return math.dist(positions[i], positions[goal])

        open_heap = [(heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        came_from = {}
        closed = set()
        while open_heap:
            _, g, node = heapq.heappop(open_heap)
            if node in closed:
                continue
            closed.add(node)
            if node == goal:
                path = [node]
                while node in came_from:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
                return path
            for neighbor, weight in graph[node]:
                new_g = g + weight
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    came_from[neighbor] = node
                    heapq.heappush(open_heap, (new_g + heuristic(neighbor), new_g, neighbor))
        return [start, goal]  # shouldn't happen for a connected roadmap; fail-safe direct hop

    def _build_route_bcd_astar(self, start_position, paint_can, redip_distance):
        """Obstacle-aware boustrophedon via classic Boustrophedon Cellular Decomposition:
        flood-fill the grid into connected regions (an opening splits a row -- and the wall --
        into separate regions), sweep each region boustrophedon-style, and route BETWEEN
        regions with A* over the (row, col) grid itself -- forced to stay on valid cells,
        detouring around whatever opening split them apart, not a straight hop across it."""
        if not hasattr(self.wall, "boustrophedon_regions") or not hasattr(self.wall, "grid_astar"):
            raise ValueError("The 'bcd_astar' strategy requires a continuous wall map (e.g. "
                              "CoordWallMap) that exposes boustrophedon_regions()/grid_astar().")
        regions = self.wall.boustrophedon_regions()
        # Only still-unpainted cells matter for this pass, so replans only re-sweep the gaps.
        regions = [[cell for cell in region if not self.wall.is_painted(*cell)] for region in regions]
        regions = [region for region in regions if region]

        route = []
        current_position = list(start_position)
        current_cell = None
        distance_since_dip = 0.0
        dip_position = paint_can.get_dip_position()

        for region in regions:
            # First region: go straight to its first cell, same as every other strategy's
            # first move. Later regions route through the grid via A* (minus both endpoints,
            # since the start is already visited and the goal is the region's own first cell).
            transit = [] if current_cell is None else self.wall.grid_astar(current_cell, region[0])[1:-1]
            for row, col in transit + region:
                position = self.wall.touch_point(row, col, offset=self.approach_offset)
                distance = math.dist(current_position, position)
                if distance_since_dip + distance > redip_distance:
                    route.append(DipStep())
                    current_position = dip_position
                    distance_since_dip = 0.0
                    distance = math.dist(current_position, position)
                route.append(CoverageTarget(row, col, position, orientation="vertical"))
                distance_since_dip += distance
                current_position = position
            current_cell = region[-1]

        return route

