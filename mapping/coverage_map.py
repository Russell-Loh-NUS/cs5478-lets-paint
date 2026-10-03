"""Coverage state over the wall grid: which cells are still unpainted."""


class CoverageMap:
    def __init__(self, wall_map):
        self.wall_map = wall_map

    def unpainted_cells(self):
        return [(row, col)
                for row in range(self.wall_map.rows)
                for col in range(self.wall_map.cols)
                if not self.wall_map.is_painted(row, col) and not self.wall_map.is_obstacle(row, col)]

    def is_complete(self):
        return self.wall_map.all_painted()