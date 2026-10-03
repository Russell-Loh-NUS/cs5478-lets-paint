"""Logical map of the wall's grid cells: which ones have been painted."""


class WallMap:
    def __init__(self, wall):
        self.wall = wall
        self.rows = wall.rows
        self.cols = wall.cols
        self._painted = [[False] * self.cols for _ in range(self.rows)]
        self._painted_obstacles = set()

    def is_obstacle(self, row, col):
        return self.wall.is_obstacle(row, col)

    def is_painted(self, row, col):
        return self._painted[row][col]

    def mark_painted(self, row, col):
        if self.is_obstacle(row, col):
            if (row, col) not in self._painted_obstacles:
                self._painted_obstacles.add((row, col))
                self.wall.flag_violation(row, col)
                print(f"[WallMap] Warning: obstacle cell ({row},{col}) was painted; counts as coverage error.")
            return
        if not self._painted[row][col]:
            self._painted[row][col] = True
            self.wall.set_cell_color(row, col, [0.0, 0.0, 1.0, 1.0])

    def painted_count(self):
        return sum(sum(r) for r in self._painted)

    def painted_obstacle_count(self):
        return len(self._painted_obstacles)

    def total_cells(self):
        """Paintable cell count (excludes obstacles like windows/doors)."""
        obstacles = sum(1 for row in range(self.rows) for col in range(self.cols)
                         if self.wall.is_obstacle(row, col))
        return self.rows * self.cols - obstacles

    def all_painted(self):
        return self.painted_count() == self.total_cells()

    def world_to_grid(self, position):
        return self.wall.world_to_grid(position)