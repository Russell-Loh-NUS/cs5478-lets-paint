"""The paint can sitting near the middle of the room: a hollow cylinder (ring wall + bottom,
open top), not a solid block -- the roller has to actually go down inside it to dip."""
import math

import pybullet as p

WALL_SEGMENTS = 16  # box "staves" approximating the ring wall (no native hollow-cylinder shape)


class PaintCan:
    def __init__(self, position=(1.5, 1.5, 0.0), inner_radius=0.10, depth=0.28,
                 wall_thickness=0.015):
        self.position = list(position)
        self.inner_radius = inner_radius
        self.depth = depth
        self.dipped = False

        outer_radius = inner_radius + wall_thickness
        angles = [2 * math.pi * i / WALL_SEGMENTS for i in range(WALL_SEGMENTS)]
        chord = 2 * outer_radius * math.sin(math.pi / WALL_SEGMENTS) * 1.05
        color = [0.05, 0.05, 0.6, 1.0]

        # Ring wall (N thin radial staves, boxes only -- mixing box + cylinder shapes in one
        # compound array crashes this pybullet build) left open at the top: genuinely hollow,
        # not just a differently colored solid cylinder, so the roller can be lowered inside.
        half_extents = [[wall_thickness / 2, chord / 2, depth / 2] for _ in range(WALL_SEGMENTS)]
        wall_positions = [[outer_radius * math.cos(a), outer_radius * math.sin(a), depth / 2]
                          for a in angles]
        wall_orientations = [p.getQuaternionFromEuler([0, 0, a]) for a in angles]

        # Visual-only prop: dipping is detected geometrically (like wall contact), so the
        # can must not physically block the arm from reaching down into it.
        wall_vis = p.createVisualShapeArray([p.GEOM_BOX] * WALL_SEGMENTS, halfExtents=half_extents,
                                             rgbaColors=[color] * WALL_SEGMENTS,
                                             visualFramePositions=wall_positions,
                                             visualFrameOrientations=wall_orientations)
        self.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=wall_vis,
                                          basePosition=[position[0], position[1], position[2]])

        # Solid floor as its own body (a lone cylinder shape is fine; it's only mixing shape
        # types in one compound array that crashes).
        bottom_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=inner_radius, length=wall_thickness,
                                          rgbaColor=color)
        self.bottom_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                            baseVisualShapeIndex=bottom_vis,
                                            basePosition=[position[0], position[1],
                                                          position[2] + wall_thickness / 2.0])

    def get_dip_position(self):
        """Center of the hollow interior (not near the rim) -- maximizes margin on both sides
        for the roller's full length to land inside once lowered."""
        return [self.position[0], self.position[1], self.position[2] + self.depth / 2.0]

    def try_dip(self, roller_a, roller_b):
        """Successful only if the ENTIRE roller -- both axis endpoints, not just one point --
        is inside the hollow interior: within the inner radius and below the rim/above the
        bottom. The two endpoints bound a straight segment through a convex region, so checking
        both suffices to guarantee the whole roller is inside."""
        bottom_z = self.position[2]
        rim_z = bottom_z + self.depth
        for x, y, z in (roller_a, roller_b):
            radial = math.hypot(x - self.position[0], y - self.position[1])
            if radial > self.inner_radius or not (bottom_z <= z <= rim_z):
                return False
        self.dipped = True
        return True
