from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass
class RollerConfig:
    radius: float = 0.035
    length: float = 0.22
    mass: float = 0.5 # Mass of roller
    max_force: float = 40.0 # To remove, once implement robot
    color: tuple = (0.85, 0.35, 0.1, 1.0)

class PaintRoller:
    def __init__(self, cfg: RollerConfig = RollerConfig(), position=(2.0, 1.5, 1.2), yaw=0.0):
        self.cfg = cfg
        r, L = cfg.radius, cfg.length
        to_y = p.getQuaternionFromEuler([math.pi / 2, 0, 0])  # cylinder z-axis -> roller y-axis
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=L,
                                     collisionFrameOrientation=to_y)
        vis = p.createVisualShapeArray(
            [p.GEOM_CYLINDER, p.GEOM_BOX], radii=[r, 0], lengths=[L, 0],
            halfExtents=[[0, 0, 0], [0.09, 0.005, 0.005]],
            visualFramePositions=[[0, 0, 0], [-0.09, 0, 0]],
            visualFrameOrientations=[to_y, [0, 0, 0, 1]],
            rgbaColors=[cfg.color, (0.2, 0.2, 0.2, 1)])
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        self.id = p.createMultiBody(cfg.mass, col, vis, basePosition=list(position),
                                    baseOrientation=orn)
        p.changeDynamics(self.id, -1, lateralFriction=0.3, linearDamping=0.5, angularDamping=0.5)
        self._cid = p.createConstraint(self.id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                       [0, 0, 0], list(position), childFrameOrientation=[0, 0, 0, 1],
                                       parentFrameOrientation=orn)
        self.set_target(position, yaw)

    def set_target(self, position, yaw):
        p.changeConstraint(self._cid, list(position), p.getQuaternionFromEuler([0, 0, yaw]),
                           maxForce=self.cfg.max_force)

    def pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        return np.array(pos), orn

    def rotation(self):
        _, orn = self.pose()
        return np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3) # Columns - x, y, z

    def axis_segment(self): # Returns the two ends of the roller
        pos, _ = self.pose()
        a = self.rotation()[:, 1] * self.cfg.length / 2
        return pos - a, pos + a
