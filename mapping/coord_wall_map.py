"""Adapter over a simulation/paint_wall.py Wall (continuous coordinate-based paint canvas) that
exposes the same interface our grid-based coverage/motion planning already expects (rows/cols/
touch_point/is_painted/is_obstacle/...). A virtual grid (sized to the roller footprint) is used
only for planning; "painted" status is read back from the wall's real paint canvas, not a
separate flag, so coverage reflects the actual continuous paint contact.
"""
import heapq
import math
import random


class CoordWallMap:
    def __init__(self, wall, cell_width, cell_height, paint_threshold=0.9, obstacle_threshold=0.5,
                 touch_spacing=None, line_spacing=None, roller_radius=0.06,
                 roller_length=0.18, floor_clearance=0.02, edge_clearance=0.01):
        self.wall = wall
        self.body_id = wall.body_id
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.paint_threshold = paint_threshold
        self.cols = max(1, int(math.ceil(wall.length / cell_width)))
        self.rows = max(1, int(math.ceil(wall.height / cell_height)))
        # Keep the physical roller above the floor and clear of perpendicular walls.
        # These are tool dimensions, independent of an optional coarse planning grid.
        self.min_u = roller_radius + edge_clearance
        self.max_u = wall.length - self.min_u
        self.min_v = roller_length / 2 + floor_clearance
        self.max_v = wall.height - roller_length / 2 - edge_clearance
        if self.min_u > self.max_u or self.min_v > self.max_v:
            raise ValueError("Wall is too small for the roller and requested clearances")
        # Sample/line spacing for the continuous "sweep" strategy -- defaults to the grid cell
        # size, but callers can pass the roller's true (unfloored) footprint for tighter,
        # genuinely continuous coverage independent of the (possibly coarser) planning grid.
        self.touch_spacing = touch_spacing or cell_width
        self.line_spacing = line_spacing or cell_height
        self._obstacles = self._compute_obstacles(obstacle_threshold)

    def _cell_uv_bounds(self, row, col):
        u0 = col * self.cell_width
        u1 = min(u0 + self.cell_width, self.wall.length)
        v0 = row * self.cell_height
        v1 = min(v0 + self.cell_height, self.wall.height)
        return u0, u1, v0, v1

    def _cell_pixel_slice(self, row, col):
        canvas = self.wall.canvas
        u0, u1, v0, v1 = self._cell_uv_bounds(row, col)
        # uv_to_px mirrors u and v, so just take both corners and sort rather than assume order.
        xa, ya = canvas.uv_to_px(u0, v0)
        xb, yb = canvas.uv_to_px(u1, v1)
        col_lo, col_hi = sorted((int(round(xa)), int(round(xb))))
        row_lo, row_hi = sorted((int(round(ya)), int(round(yb))))
        col_lo, col_hi = max(0, col_lo), min(canvas.width_px, col_hi)
        row_lo, row_hi = max(0, row_lo), min(canvas.height_px, row_hi)
        return row_lo, row_hi, col_lo, col_hi

    def _compute_obstacles(self, obstacle_threshold):
        obstacles = set()
        canvas = self.wall.canvas
        for row in range(self.rows):
            for col in range(self.cols):
                r0, r1, c0, c1 = self._cell_pixel_slice(row, col)
                region = canvas.paintable[r0:r1, c0:c1]
                if region.size == 0 or region.mean() < obstacle_threshold:
                    obstacles.add((row, col))
        return obstacles

    def is_obstacle(self, row, col):
        return (row, col) in self._obstacles

    def cell_painted_fraction(self, row, col):
        if self.is_obstacle(row, col):
            return 0.0
        r0, r1, c0, c1 = self._cell_pixel_slice(row, col)
        region = self.wall.canvas.painted[r0:r1, c0:c1]
        if region.size == 0:
            return 0.0
        paintable = self.wall.canvas.paintable[r0:r1, c0:c1]
        if not paintable.any():
            return 0.0
        return float(region[paintable].mean())

    def is_painted(self, row, col):
        return self.cell_painted_fraction(row, col) >= self.paint_threshold

    def mark_painted(self, row, col):
        """No-op: painting happens physically via roller contact (CoordRobotController), not
        explicit marking -- this only exists so the shared CoverageMap/planner API is satisfied.
        """

    def painted_count(self):
        return sum(1 for r in range(self.rows) for c in range(self.cols)
                   if not self.is_obstacle(r, c) and self.is_painted(r, c))

    def painted_obstacle_count(self):
        # Obstacles are excluded from the canvas's own "painted" mask, so they can never
        # register as painted -- there is nothing to count as a violation.
        return 0

    def total_cells(self):
        return sum(1 for r in range(self.rows) for c in range(self.cols) if not self.is_obstacle(r, c))

    def all_painted(self):
        return self.painted_count() == self.total_cells()

    def world_to_grid(self, position):
        u, v = self.wall.world_to_uv(position)
        col = int(u // self.cell_width)
        row = int(v // self.cell_height)
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return row, col

    def touch_point(self, row, col, offset=0.05):
        """World position just in front of the wall face, for the roller to approach."""
        u0, u1, v0, v1 = self._cell_uv_bounds(row, col)
        u, v = self._safe_uv((u0 + u1) / 2.0, (v0 + v1) / 2.0)
        world = self.wall.uv_to_world(u, v)
        return (world + self.wall.normal * offset).tolist()

    def _safe_uv(self, u, v):
        return (max(self.min_u, min(self.max_u, u)),
                max(self.min_v, min(self.max_v, v)))

    def cell_center(self, row, col):
        u0, u1, v0, v1 = self._cell_uv_bounds(row, col)
        return self.wall.uv_to_world((u0 + u1) / 2.0, (v0 + v1) / 2.0).tolist()

    def _in_opening(self, u, v):
        return any(o.u <= u <= o.u + o.width and o.v <= v <= o.v + o.height for o in self.wall.openings)

    def continuous_targets(self, offset=0.05):
        """Continuous boustrophedon sweep directly over the wall's real (u, v) surface, with
        sample/line spacing from touch_spacing/line_spacing (the roller's true footprint) --
        not the planning grid. Skips samples that fall inside a door/window opening, and
        samples whose grid cell is already painted (so replans only re-sweep what's left).
        Returns an ordered list of (row, col, world_position) tuples for the "sweep" strategy.
        """
        targets = []
        v = self.line_spacing / 2.0
        line_index = 0
        while v < self.wall.height:
            u_values = []
            u = self.touch_spacing / 2.0
            while u < self.wall.length:
                if not self._in_opening(u, v):
                    u_values.append(u)
                u += self.touch_spacing
            if line_index % 2 == 1:
                u_values.reverse()
            for u in u_values:
                safe_u, safe_v = self._safe_uv(u, v)
                if self._in_opening(safe_u, safe_v):
                    continue
                on_wall = self.wall.uv_to_world(safe_u, safe_v)
                row, col = self.world_to_grid(on_wall)
                if self.is_painted(row, col):
                    continue
                position = (on_wall + self.wall.normal * offset).tolist()
                targets.append((row, col, position))
            v += self.line_spacing
            line_index += 1
        return targets

    def prm_samples(self, count=None, offset=0.05, rng=None):
        """Randomly sample valid (non-opening, not-yet-painted) points across the wall's real
        surface for the "prm" coverage strategy's roadmap -- sample count defaults to roughly
        one per roller footprint, so the roadmap is dense enough to plausibly cover the wall.
        Returns a list of (row, col, world_position) tuples, like continuous_targets().
        """
        rng = rng or random.Random()
        if count is None:
            area = self.wall.length * self.wall.height
            footprint = self.touch_spacing * self.line_spacing
            count = max(4, int(area / footprint))
        samples = []
        attempts = 0
        while len(samples) < count and attempts < count * 20:
            attempts += 1
            u = rng.uniform(self.min_u, self.max_u)
            v = rng.uniform(self.min_v, self.max_v)
            if self._in_opening(u, v):
                continue
            on_wall = self.wall.uv_to_world(u, v)
            row, col = self.world_to_grid(on_wall)
            if self.is_painted(row, col):
                continue
            position = (on_wall + self.wall.normal * offset).tolist()
            samples.append((row, col, position))
        return samples

    def boustrophedon_regions(self):
        """Cellular decomposition for the "bcd_astar" strategy: flood-fill the (row, col) grid
        into connected non-obstacle regions (4-connectivity) -- an opening that splits a row
        in two creates two separate regions, matching the classic Boustrophedon Cellular
        Decomposition approach. Returns a list of regions, each an ordered list of (row, col)
        cells swept boustrophedon-style (row by row, alternating direction) within that region.
        """
        visited = set()
        regions = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) in visited or self.is_obstacle(row, col):
                    continue
                component = set()
                queue = [(row, col)]
                visited.add((row, col))
                while queue:
                    r, c = queue.pop()
                    component.add((r, c))
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if (0 <= nr < self.rows and 0 <= nc < self.cols and
                                (nr, nc) not in visited and not self.is_obstacle(nr, nc)):
                            visited.add((nr, nc))
                            queue.append((nr, nc))

                by_row = {}
                for r, c in component:
                    by_row.setdefault(r, []).append(c)
                ordered = []
                for i, r in enumerate(sorted(by_row)):
                    cols_in_row = sorted(by_row[r])
                    if i % 2 == 1:
                        cols_in_row.reverse()
                    for c in cols_in_row:
                        ordered.append((r, c))
                regions.append(ordered)
        return regions

    def grid_astar(self, start, goal):
        """A* shortest path over the (row, col) grid for the "bcd_astar" strategy's inter-region
        transitions -- treats obstacle cells as blocked (4-connectivity), so the path is forced
        to stay on valid cells and detour around whatever opening split the regions apart,
        instead of just hopping directly across it like the other continuous strategies do."""
        def heuristic(rc):
            return math.dist(rc, goal)

        open_heap = [(heuristic(start), 0, start)]
        g_score = {start: 0}
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
            r, c = node
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.is_obstacle(nr, nc):
                    continue
                neighbor = (nr, nc)
                new_g = g + 1
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    came_from[neighbor] = node
                    heapq.heappush(open_heap, (new_g + heuristic(neighbor), new_g, neighbor))
        return [start]  # unreachable (shouldn't happen for a single-wall grid); fail-safe
