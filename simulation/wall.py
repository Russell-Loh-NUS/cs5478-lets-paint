"""A paintable wall: one solid collision body plus a visual grid overlay sized to the roller.

Generalized to any of the 4 room-edge orientations, so all 4 walls of a room can be built the
same way: axis='y' for the back/front walls (which run along world x, at a fixed world y),
axis='x' for the left/right walls (which run along world y, at a fixed world x).
"""

import pybullet as p

UNPAINTED_COLOR = [0.85, 0.85, 0.8, 1.0]
PAINTED_COLOR = [0.0, 0.0, 1.0, 1.0]
OBSTACLE_COLOR = [0.55, 0.35, 0.15, 1.0] # non-paintable (e.g. a window/door frame)
VIOLATION_COLOR = [1.0, 0.0, 0.0, 1.0] # an obstacle that was painted anyway


class Wall:
    def __init__(self, axis, position, length=10.0, height=10.0, grid_width=0.1, grid_height=0.5,
                thickness=0.1, name="wall"):
        """axis: 'x' or 'y', the world axis this wall is perpendicular to (its fixed coordinate).
        position: the world coordinate of the wall's inner face along that axis (e.g. +-5, a
        room edge). length: the wall's span along the other horizontal axis."""
        self.axis = axis
        self.position = position
        self.length = length
        self.height = height
        self.name = name
        # Each grid cell matches one roller "stamp": grid_width ~ roller diameter,
        # grid_height ~ roller length (for a vertically-held roller).
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.thickness = thickness
        self.cols = int(round(length / grid_width))
        self.rows = int(round(height / grid_height))
        # Which way is "into the room" from this wall's face (room is centered at the origin).
        self.inward_sign = -1.0 if position > 0 else 1.0
        # Backward-compatible alias: the fixed coordinate, as used by code written for the
        # original single (axis='y') wall.
        self.inner_face_y = position

        self.body_id = self._create_wall_body()
        self.cell_ids = self._create_grid_overlay()
        self.obstacle_cells = set()
        self.visible = True

    def _axis_point(self, offset):
        """World coordinate on the wall's fixed axis, offset `offset` further into the room."""
        return self.position + self.inward_sign * offset

    def _to_world(self, u, z, offset):
        fixed = self._axis_point(offset)
        return [u, fixed, z] if self.axis == "y" else [fixed, u, z]

    def _create_wall_body(self):
        thin, wide = self.thickness / 2.0, self.length / 2.0
        half_extents = [wide, thin, self.height / 2.0] if self.axis == "y" else [thin, wide, self.height / 2.0]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1.0, 1.0, 1.0, 1.0])
        base_position = self._to_world(0.0, self.height / 2.0, -self.thickness / 2.0)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape,
                                basePosition=base_position)

    def _create_grid_overlay(self):
        # Shared visual shape for every cell; each cell gets its own (collision-free) body
        # so its color can be toggled independently when painted.
        half_u = self.grid_width / 2.0 * 0.95
        half_z = self.grid_height / 2.0 * 0.95
        half_extents = [half_u, 0.005, half_z] if self.axis == "y" else [0.005, half_u, half_z]
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=UNPAINTED_COLOR)
        cell_ids = [[None] * self.cols for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                u, z = self._cell_center_uz(row, col)
                body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_shape, basePosition=self._to_world(u, z, 0.02))
                cell_ids[row][col] = body_id
        return cell_ids

    def _cell_center_uz(self, row, col):
        u = -self.length / 2.0 + self.grid_width * (col + 0.5)
        z = self.grid_height * (row + 0.5)
        return u, z

    def cell_center(self, row, col):
        u, z = self._cell_center_uz(row, col)
        return self._to_world(u, z, 0.0)

    def touch_point(self, row, col, offset=0.05):
        """World position just in front of the wall face, for the roller to approach."""
        u, z = self._cell_center_uz(row, col)
        return self._to_world(u, z, offset)

    def world_to_grid(self, position):
        u = position[0] if self.axis == "y" else position[1]
        z = position[2]
        col = int((u + self.length / 2.0) / self.grid_width)
        row = int(z / self.grid_height)
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return row, col

    def mark_obstacle(self, row, col):
        """Flag a grid cell as a non-paintable obstacle (e.g. a window or door)."""
        self.obstacle_cells.add((row, col))
        self.set_cell_color(row, col, OBSTACLE_COLOR)

    def is_obstacle(self, row, col):
        return (row, col) in self.obstacle_cells

    def flag_violation(self, row, col):
        """Visually flag that a non-paintable obstacle cell was painted anyway."""
        self.set_cell_color(row, col, VIOLATION_COLOR)

    def set_cell_color(self, row, col, rgba):
        p.changeVisualShape(self.cell_ids[row][col], -1, rgbaColor=rgba)

    def set_visible(self, visible):
        """Show/hide the whole wall (body + grid), preserving each cell's current color."""
        self.visible = visible
        alpha = 1.0 if visible else 0.0
        self._set_alpha(self.body_id, alpha)
        for row in range(self.rows):
            for col in range(self.cols):
                self._set_alpha(self.cell_ids[row][col], alpha)

    def _set_alpha(self, body_id, alpha):
        data = p.getVisualShapeData(body_id)
        if not data:
            return
        r, g, b, _ = data[0][7]
        p.changeVisualShape(body_id, -1, rgbaColor=[r, g, b, alpha])