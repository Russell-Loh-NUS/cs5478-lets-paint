"""Coverage state over the wall grid: which cells are still unpainted."""


class CoverageMap:
    def __init__(self, wall_map):
        self.wall_map = wall_map

    def unpainted_cells(self, threshold=0.8):
        return [(row, col)
                for row in range(self.wall_map.rows)
                for col in range(self.wall_map.cols)
                if not self.wall_map.is_obstacle(row, col)
                and self.wall_map.cell_painted_fraction(row, col) < threshold]

    def is_complete(self, threshold=0.8):
        return not self.unpainted_cells(threshold=threshold)

