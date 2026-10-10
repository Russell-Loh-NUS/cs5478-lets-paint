"""Coverage planner: plans the entire paint route up front, with four selectable strategies.

- "nearest_neighbor": greedily visits the nearest remaining unpainted cell each step. Not a
  provably optimal TSP tour, but it naturally clusters nearby cells instead of forcing a rigid
  row-by-row sweep. Redips only when forced (budget about to run out).
- "lawnmower": the classic fixed boustrophedon sweep (row by row, alternating direction).
  Redips only when forced, same as above.
- "optimized": builds a nearest-neighbor tour and then improves it with 2-opt local search (so
  the visiting order itself approximates the shortest path over the whole wall, not just a
  greedy walk). Then, for that fixed order, a dynamic program chooses the cost-minimizing set
  of dip stops: dipping is allowed at any point in the route (not just when forced), and is
  only ever inserted where it reduces -- or is required to achieve -- total path length,
  subject to never exceeding redip_distance of travel since the last dip.
- "astar": solves the visiting order exactly via A* search over (current cell, visited-set)
  states, with an admissible nearest-unvisited-cell heuristic -- the provably shortest possible
  Hamiltonian path, not just a heuristic approximation. The state space is exponential in the
  cell count (2^n) and ALWAYS runs exactly -- no fallback to a heuristic tour -- so it must be
  given a small enough grid (tens of cells, not hundreds) by the caller or it will never
  finish. Dip placement reuses the same optimal DP as "optimized".
- "sweep": continuous boustrophedon coverage directly over the wall's real (u, v) surface
  (see CoordWallMap.continuous_targets), with sample/line spacing from the roller's actual
  footprint rather than the (possibly coarser) planning grid used by the other strategies.
  Visiting order is the sweep itself, not a TSP tour. Only available for a continuous wall
  map (one that exposes continuous_targets(), e.g. CoordWallMap) -- the legacy grid-only Wall
  doesn't support it.
- "grid_smooth": continuous A* via option 1 (discretize + smooth) -- takes the same fine
  boustrophedon grid as "sweep", then repeatedly applies Chaikin corner-cutting to round off
  its sharp right-angle turns into a smooth curve, instead of a blocky grid-aligned path.
- "prm": continuous A* via option 2 (sampling-based) -- randomly samples valid points across
  the wall (a probabilistic roadmap), connects each to its K nearest neighbors into a graph,
  picks an overall visiting order with the same nearest-neighbor + 2-opt heuristic as
  "optimized", then routes BETWEEN consecutive targets with A* over the roadmap graph (not a
  straight line) -- since the roadmap only connects nearby samples, distant targets may need
  several hops, found exactly via graph search rather than guessed. Dipping is still only
  forced when the budget is about to run out (like "nearest_neighbor"/"lawnmower"), not placed
  optimally -- see "optimized_prm" for that.
- "optimized_prm": "optimized" fully adapted to continuous PRM coverage -- same roadmap,
  same nearest-neighbor + 2-opt visiting order, and same A*-routed hops as "prm", but dipping
  also reuses "optimized"'s DP-optimal placement (free to land anywhere it reduces total
  travel, not just when forced) over the full A*-expanded waypoint sequence.
- "bcd_astar": obstacle-aware boustrophedon via classic Boustrophedon Cellular Decomposition --
  flood-fills the grid into connected regions split by openings (see
  CoordWallMap.boustrophedon_regions), sweeps each region boustrophedon-style, then routes
  BETWEEN regions with A* over the (row, col) grid itself (see CoordWallMap.grid_astar),
  forced to stay on valid cells rather than hopping straight across the gap. "sweep"/
  "grid_smooth" already skip obstacle points individually but don't decompose into regions or
  route transitions this way -- though since openings are just empty airspace in front of the
  wall (not physical obstacles blocking the robot's 3D path), a straight hop across one is
  already perfectly safe here, so this mostly matters for genuinely blocking obstacles.

All nine respect the same hard dip-distance budget, so they can be run back-to-back and
compared (total path length, redip count, etc.) on equal footing.
"""
import heapq
import math


class CoverageTarget:
    def __init__(self, row, col, position, orientation):
        self.row = row
        self.col = col
        self.position = position
        self.orientation = orientation


class DipStep:
    """A marker in the planned route: redip before continuing to the next target."""


class CoveragePlanner:
    STRATEGIES = ("nearest_neighbor", "lawnmower", "optimized", "astar", "sweep", "grid_smooth",
                  "prm", "optimized_prm", "bcd_astar")
    # How many nearest neighbors each PRM sample connects to in the roadmap graph.
    PRM_K_NEIGHBORS = 6

    def __init__(self, coverage_map, wall, approach_offset=0.05):
        self.coverage_map = coverage_map
        self.wall = wall
        self.approach_offset = approach_offset

    def plan_full_route(self, start_position, paint_can, redip_distance, strategy="nearest_neighbor"):
        """Read the whole unpainted set and return a dip-aware route, using the given strategy."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown coverage strategy '{strategy}', expected one of {self.STRATEGIES}")

        if strategy == "sweep":
            return self._build_route_sweep(start_position, paint_can, redip_distance)
        if strategy == "grid_smooth":
            return self._build_route_grid_smooth(start_position, paint_can, redip_distance)
        if strategy == "prm":
            return self._build_route_prm(start_position, paint_can, redip_distance)
        if strategy == "optimized_prm":
            return self._build_route_optimized_prm(start_position, paint_can, redip_distance)
        if strategy == "bcd_astar":
            return self._build_route_bcd_astar(start_position, paint_can, redip_distance)

        remaining = self.coverage_map.unpainted_cells(threshold=0.8)
        cell_positions = {(row, col): self.wall.touch_point(row, col, offset=self.approach_offset)
                           for row, col in remaining}

        if strategy == "lawnmower":
            ordered_cells = self._lawnmower_order(remaining)
            return self._build_route_fixed_order(ordered_cells, cell_positions, start_position,
                                                   paint_can, redip_distance)
        if strategy == "optimized":
            order = self._nearest_neighbor_order(remaining, cell_positions, start_position)
            order = self._two_opt_improve(order, cell_positions, start_position)
            return self._insert_optimal_dips(order, cell_positions, start_position, paint_can, redip_distance)
        if strategy == "astar":
            order = self._astar_order(remaining, cell_positions, start_position)
            return self._insert_optimal_dips(order, cell_positions, start_position, paint_can, redip_distance)
        if strategy == "nearest_neighbor":
            return self._build_route_nearest_neighbor(remaining, cell_positions, start_position,
                                                        paint_can, redip_distance)
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

    def _lawnmower_order(self, remaining):
        # Sweep each row left-to-right, then right-to-left on the next row, bottom to top.
        remaining_set = set(remaining)
        ordered = []
        for row in range(self.wall.rows):
            cols = range(self.wall.cols) if row % 2 == 0 else reversed(range(self.wall.cols))
            for col in cols:
                if (row, col) in remaining_set:
                    ordered.append((row, col))
        return ordered

    def _nearest_neighbor_order(self, remaining, cell_positions, start_position):
        """Greedy nearest-neighbor visiting order, ignoring dip stops (seed tour for 2-opt)."""
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

    def _astar_order(self, remaining, cell_positions, start_position):
        """Exact shortest visiting order via A* search over (current cell, visited-set) states.
        Provably optimal, unlike the 2-opt heuristic -- but the state space is exponential in
        the number of cells, so this always runs for real with no fallback; it's the caller's
        job to keep the grid small enough (see main_lets_paint.py's astar-specific cell size).
        """
        cells = list(remaining)
        n = len(cells)
        if n == 0:
            return []

        positions = [cell_positions[c] for c in cells]
        full_mask = (1 << n) - 1

        def heuristic(position, mask):
            # Admissible: at least this far must still be travelled to reach any unvisited cell.
            unvisited_distances = [math.dist(position, positions[i]) for i in range(n) if not (mask & (1 << i))]
            return min(unvisited_distances) if unvisited_distances else 0.0

        start_state = (-1, 0)  # (current cell index, visited bitmask); -1 = at start_position
        g_score = {start_state: 0.0}
        came_from = {}
        open_heap = [(heuristic(start_position, 0), 0.0, start_state)]
        closed = set()

        while open_heap:
            _, g, state = heapq.heappop(open_heap)
            if state in closed:
                continue
            closed.add(state)
            node, mask = state

            if mask == full_mask:
                order_indices = []
                current = state
                while current in came_from:
                    previous, added_index = came_from[current]
                    order_indices.append(added_index)
                    current = previous
                order_indices.reverse()
                return [cells[i] for i in order_indices]

            current_position = start_position if node == -1 else positions[node]
            for next_index in range(n):
                if mask & (1 << next_index):
                    continue
                new_mask = mask | (1 << next_index)
                new_state = (next_index, new_mask)
                new_g = g + math.dist(current_position, positions[next_index])
                if new_g < g_score.get(new_state, float("inf")):
                    g_score[new_state] = new_g
                    came_from[new_state] = (state, next_index)
                    new_f = new_g + heuristic(positions[next_index], new_mask)
                    heapq.heappush(open_heap, (new_f, new_g, new_state))

        return cells  # unreachable for a complete graph, but avoids crashing if it ever is

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

    def _build_route_nearest_neighbor(self, remaining, cell_positions, start_position, paint_can, redip_distance):
        remaining = list(remaining)
        route = []
        current_position = list(start_position)
        distance_since_dip = 0.0
        dip_position = paint_can.get_dip_position()

        while remaining:
            best_cell, best_distance = None, None
            for cell in remaining:
                distance = math.dist(current_position, cell_positions[cell])
                if best_distance is None or distance < best_distance:
                    best_cell, best_distance = cell, distance

            if distance_since_dip + best_distance > redip_distance:
                route.append(DipStep())
                current_position = dip_position
                distance_since_dip = 0.0
                best_distance = math.dist(current_position, cell_positions[best_cell])

            remaining.remove(best_cell)
            row, col = best_cell
            route.append(CoverageTarget(row, col, cell_positions[best_cell], orientation="vertical"))
            distance_since_dip += best_distance
            current_position = cell_positions[best_cell]

        return route

    def _build_route_sweep(self, start_position, paint_can, redip_distance):
        """Continuous boustrophedon sweep directly over the wall's real surface (see
        CoordWallMap.continuous_targets) -- the visiting order IS the sweep, so there's no
        TSP ordering step here, just dip insertion along the fixed sweep order."""
        if not hasattr(self.wall, "continuous_targets"):
            raise ValueError("The 'sweep' strategy requires a continuous wall map (e.g. "
                              "CoordWallMap) that exposes continuous_targets().")
        points = self.wall.continuous_targets()
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

    def _build_route_grid_smooth(self, start_position, paint_can, redip_distance):
        """Continuous A*, option 1 (discretize + smooth): reuse the same fine boustrophedon
        grid as "sweep", then round off its sharp right-angle turns with Chaikin corner-cutting
        so the roller follows a smooth curve instead of a blocky grid-aligned path."""
        if not hasattr(self.wall, "continuous_targets"):
            raise ValueError("The 'grid_smooth' strategy requires a continuous wall map (e.g. "
                              "CoordWallMap) that exposes continuous_targets().")
        points = self.wall.continuous_targets()
        smoothed = self._chaikin_smooth(points)
        return self._build_route_fixed_points(smoothed, start_position, paint_can, redip_distance)

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

    def _build_route_prm(self, start_position, paint_can, redip_distance):
        """Continuous A*, option 2 (sampling-based): a probabilistic roadmap (PRM). Randomly
        sample valid points on the wall, connect each to its K nearest neighbors into a graph,
        pick an overall visiting order (nearest-neighbor + 2-opt, same as "optimized"), then
        route between consecutive targets with A* over the roadmap -- not a straight line --
        since the roadmap's sparse connectivity may force a multi-hop path between them."""
        samples, positions, graph, order = self._build_prm_roadmap(start_position)
        if not samples:
            return []

        route = []
        current_position = list(start_position)
        current_node = None
        distance_since_dip = 0.0
        dip_position = paint_can.get_dip_position()

        for node in order:
            # First hop has no roadmap node to start from yet -- go straight to it, same as
            # every other strategy's first move. Later hops route through the roadmap via A*.
            hop_nodes = [node] if current_node is None else self._roadmap_astar(graph, positions, current_node, node)[1:]
            for idx in hop_nodes:
                position = positions[idx]
                distance = math.dist(current_position, position)
                if distance_since_dip + distance > redip_distance:
                    route.append(DipStep())
                    current_position = dip_position
                    distance_since_dip = 0.0
                    distance = math.dist(current_position, position)
                row, col, _ = samples[idx]
                route.append(CoverageTarget(row, col, position, orientation="vertical"))
                distance_since_dip += distance
                current_position = position
            current_node = node

        return route

    def _build_route_optimized_prm(self, start_position, paint_can, redip_distance):
        """"optimized" fully adapted to continuous PRM coverage: same roadmap, same
        nearest-neighbor + 2-opt visiting order, and same A*-routed hops as "prm" -- but
        dipping also reuses "optimized"'s DP-optimal placement over the full A*-expanded
        waypoint sequence, instead of "prm"'s simple forced-only redipping."""
        samples, positions, graph, order = self._build_prm_roadmap(start_position)
        if not samples:
            return []

        ordered_points = []
        current_node = None
        for node in order:
            hop_nodes = [node] if current_node is None else self._roadmap_astar(graph, positions, current_node, node)[1:]
            for idx in hop_nodes:
                row, col, _ = samples[idx]
                ordered_points.append((row, col, positions[idx]))
            current_node = node

        return self._insert_optimal_dips_points(ordered_points, start_position, paint_can, redip_distance)

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
        order = self._nearest_neighbor_order(list(range(n)), index_positions, start_position)
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

