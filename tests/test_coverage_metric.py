import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapping.coord_wall_map import CoordWallMap
from simulation.paint_canvas import PaintCanvas


class DummyWall:
    def __init__(self, length=1.0, height=1.0):
        self.length = length
        self.height = height
        self.origin = np.zeros(3)
        self.u_dir = np.array([1.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, 1.0])
        self.body_id = 1
        self.openings = ()
        self.canvas = PaintCanvas(length, height, px_per_m=100, base_rgb=(0, 0, 0), paint_rgb=(255, 0, 0))


class CoverageMetricTest(unittest.TestCase):
    def test_partial_cell_is_not_treated_as_fully_covered(self):
        wall = DummyWall()
        wall_map = CoordWallMap(wall, cell_width=0.5, cell_height=0.5)

        # Paint about 75% of the cell to mimic a partially covered patch; the default threshold
        # should not let that count as a fully covered cell.
        paint_width = int(round(0.75 * 50))
        wall.canvas._mask.paste(255, (0, 0, paint_width, 50))

        self.assertLess(wall_map.cell_painted_fraction(0, 0), 0.9)
        self.assertFalse(wall_map.is_painted(0, 0))


if __name__ == "__main__":
    unittest.main()
