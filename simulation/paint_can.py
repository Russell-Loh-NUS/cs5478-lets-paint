"""A fixed, collidable paint can with a hollow interior and an open top."""
import math

import pybullet as p

WALL_SEGMENTS = 16


class PaintCan:
    def __init__(self, position=(1.5, 1.5, 0.0), inner_radius=0.10, depth=0.28,
                 wall_thickness=0.015):
        self.position = list(position)
        self.inner_radius = inner_radius
        self.outer_radius = inner_radius + wall_thickness
        self.depth = depth
        self.wall_thickness = wall_thickness
        self.dipped = False
        if inner_radius <= 0 or not 0 < wall_thickness < depth:
            raise ValueError("Invalid can dimensions")

        # Radially oriented boxes form a solid ring, NOT a solid cylinder.
        # Each inner face is exactly inner_radius from the can centre.
        mid_radius = inner_radius + wall_thickness / 2
        angles = [2 * math.pi * i / WALL_SEGMENTS for i in range(WALL_SEGMENTS)]
        tangent_half_width = self.outer_radius * math.tan(math.pi / WALL_SEGMENTS)
        half_extents = [[wall_thickness / 2, tangent_half_width, depth / 2]
                        for _ in angles]
        positions = [[mid_radius * math.cos(a), mid_radius * math.sin(a), depth / 2]
                     for a in angles]
        orientations = [p.getQuaternionFromEuler([0, 0, a]) for a in angles]
        color = [0.05, 0.05, 0.6, 1.0]
        collision = p.createCollisionShapeArray(
            [p.GEOM_BOX] * WALL_SEGMENTS, halfExtents=half_extents,
            collisionFramePositions=positions, collisionFrameOrientations=orientations)
        visual = p.createVisualShapeArray(
            [p.GEOM_BOX] * WALL_SEGMENTS, halfExtents=half_extents,
            visualFramePositions=positions, visualFrameOrientations=orientations,
            rgbaColors=[color] * WALL_SEGMENTS)
        self.body_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual, basePosition=self.position)

        bottom_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self.outer_radius, height=wall_thickness)
        bottom_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=self.outer_radius, length=wall_thickness, rgbaColor=color)
        self.bottom_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=bottom_collision,
            baseVisualShapeIndex=bottom_visual,
            basePosition=[position[0], position[1], position[2] + wall_thickness / 2])
        self.collision_body_ids = (self.body_id, self.bottom_id)

    def get_dip_position(self):
        """Centre of the usable space ABOVE the solid bottom."""
        return [self.position[0], self.position[1],
                self.position[2] + (self.wall_thickness + self.depth) / 2]

    def get_approach_position(self, roller_length=0.18, roller_radius=0.06, clearance=0.05):
        """Whole roller clears the rim, including small orientation deviations."""
        extent = math.hypot(roller_length / 2, roller_radius)
        return [self.position[0], self.position[1], self.position[2] + self.depth + extent + clearance]

    def try_dip(self, roller_a, roller_b, roller_radius=0.06):
        """Check the full barrel, not just its centreline, against the usable interior."""
        length = math.dist(roller_a, roller_b)
        if length <= 1e-12 or roller_radius <= 0:
            return False
        axis_z = min(1.0, abs(roller_a[2] - roller_b[2]) / length)
        radial_z = roller_radius * math.sqrt(max(0.0, 1 - axis_z * axis_z))
        bottom = self.position[2] + self.wall_thickness
        rim = self.position[2] + self.depth
        margin = 0.002
        for x, y, z in (roller_a, roller_b):
            radial = math.hypot(x - self.position[0], y - self.position[1])
            if (radial + roller_radius > self.inner_radius - margin or
                    z - radial_z < bottom + margin or z + radial_z > rim - margin):
                return False
        self.dipped = True
        return True
