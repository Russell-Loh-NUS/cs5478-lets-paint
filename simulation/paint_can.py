"""The paint can sitting near the middle of the room."""
import pybullet as p


class PaintCan:
    def __init__(self, position=(1.5, 1.5, 0.0), radius=0.5, height=0.5, dip_threshold=0.5):
        self.position = list(position)
        self.radius = radius
        self.height = height
        self.dip_threshold = dip_threshold
        self.dipped = False

        base_z = position[2] + height / 2.0
        # Visual-only prop: dipping is detected geometrically (like wall contact), so the
        # can must not physically block the arm from reaching down into it.
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                        rgbaColor=[0.05, 0.05, 0.6, 1.0])
        self.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                            baseVisualShapeIndex=vis_shape,
                                            basePosition=[position[0], position[1], base_z])

    def get_dip_position(self):
        return [self.position[0], self.position[1], self.position[2] + self.height * 0.8]

    def distance_to(self, position):
        cx, cy, cz = self.position[0], self.position[1], self.position[2] + self.height / 2.0
        return ((position[0] - cx) ** 2 + (position[1] - cy) ** 2 + (position[2] - cz) ** 2) ** 0.5

    def try_dip(self, roller_position):
        if self.distance_to(roller_position) <= self.dip_threshold:
            self.dipped = True
            return True
        return False