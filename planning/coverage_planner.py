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
cell count (2^n), so this is only run for small instances (see ASTAR_MAX_CELLS); larger
instances fall back to the "optimized" tour instead of hanging. Dip placement reuses the
same optimal DP as "optimized".

All four respect the same hard dip-distance budget, so they can be run back-to-back and
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
    STRATEGIES = ("nearest_neighbor", "lawnmower", "optimized", "astar")
    # A* state space is 2^n; beyond this many cells it's no longer practical to run exactly.
    ASTAR_MAX_CELLS = 13

    def __init__(self, coverage_map, wall, approach_offset=0.05):
        self.coverage_map = coverage_map
        self.wall = wall
        self.approach_offset = approach_offset

    def plan_full_route(self, start_position, paint_can, redip_distance, strategy="nearest_neighbor"):
        """Read the whole unpainted set and return a dip-aware route, using the given strategy."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown coverage strategy '{strategy}', expected one of {self.STRATEGIES}")

        remaining = self.coverage_map.unpainted_cells()
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
                print(f" {i:4d}. DIP at paint can")
            else:
                x, y, z = step.position
                print(f" {i:4d}. PAINT cell (row={step.row}, col={step.col}) "
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
        Provably optimal, unlike the 2-opt heuristic, but the state space is exponential in the
        number of cells -- falls back to nearest-neighbor + 2-opt above ASTAR_MAX_CELLS."""
        cells = list(remaining)
        n = len(cells)
        if n == 0:
            return []
        if n > self.ASTAR_MAX_CELLS:
            print(f"[CoveragePlanner] A* is exact but exponential (2^n states); {n} cells "
                f"exceeds the practical cap of {self.ASTAR_MAX_CELLS}. "
                "Falling back to nearest-neighbor + 2-opt.")
            order = self._nearest_neighbor_order(remaining, cell_positions, start_position)
            return self._two_opt_improve(order, cell_positions, start_position)

        positions = [cell_positions[c] for c in cells]
        full_mask = (1 << n) - 1

        def heuristic(position, mask):
            # Admissible: at least this far must still be travelled to reach any unvisited cell.
            unvisited_distances = [math.dist(position, positions[i]) for i in range(n) if not (mask & (1 << i))]
            return min(unvisited_distances) if unvisited_distances else 0.0

        start_state = (-1, 0) # (current cell index, visited bitmask); -1 = at start_position
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

        return cells # unreachable for a complete graph, but avoids crashing if it ever is

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
