"""Creates the room: a 10m x 10m floor and the physics world setup."""
import pybullet as p


class Environment:
    def __init__(self, gui=True, floor_size=10.0):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        # Disable rendering while building static geometry (wall grid has many bodies).
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.floor_size = floor_size
        self.floor_id = self._create_floor(floor_size)

    def _create_floor(self, size):
        half = size / 2.0
        thickness = 0.05
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, thickness])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, thickness], rgbaColor=[0.55, 0.55, 0.55, 1])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                basePosition=[0, 0, -thickness])

    def finish_setup(self):
        """Re-enable rendering once static geometry is built."""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def focus_camera_on_wall(self, wall, distance=9.0, pitch=-20.0):
        """Point the GUI camera at the given wall, whichever of the 4 room edges it is.

        Empirically verified (pybullet's yaw convention isn't the usual compass one):
        yaw=0 -> +y wall (back), 180 -> -y wall (front), 90 -> -x wall (left), -90 -> +x wall (right).
        The target stays at the room center (not offset toward the wall), which is what
        actually produces a square, correctly-framed view -- an offset target skews it badly.
        """
        if wall.axis == "y":
            yaw = 0.0 if wall.position > 0 else 180.0
        else:
            yaw = -90.0 if wall.position > 0 else 90.0
        target = [0.0, 0.0, wall.height * 0.4]
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch,
                                        cameraTargetPosition=target)

    def step(self):
        p.stepSimulation()