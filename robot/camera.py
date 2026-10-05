from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from .roller import PaintRoller


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 240
    fov: float = 90.0
    near: float = 0.05
    far: float = 6.0
    eye: tuple = (-0.30, 0.0, 0.14) # Mount position
    target: tuple = (0.50, 0.0, -0.12) # The target position for camera to look at


class PerceptionCamera:
    def __init__(self, roller: PaintRoller, cfg: CameraConfig = CameraConfig(),
                 before_capture=None):
        self.roller, self.cfg, self._before = roller, cfg, before_capture

    def capture(self): # Capture a frame of the camera view
        if self._before:
            self._before()
        c = self.cfg
        pos, _ = self.roller.pose()
        R = self.roller.rotation()
        eye = pos + R @ np.array(c.eye)
        target = pos + R @ np.array(c.target)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), R[:, 2].tolist())
        proj = p.computeProjectionMatrixFOV(c.fov, c.width / c.height, c.near, c.far)
        _, _, rgba, depth, seg = p.getCameraImage(c.width, c.height, view, proj,
                                                  renderer=p.ER_TINY_RENDERER)
        rgb = np.asarray(rgba, np.uint8).reshape(c.height, c.width, 4)[..., :3]
        d = np.asarray(depth, np.float32).reshape(c.height, c.width)
        depth_m = c.far * c.near / (c.far - (c.far - c.near) * d)
        return {"rgb": rgb, "depth": depth_m,
                "seg": np.asarray(seg, np.int32).reshape(c.height, c.width),
                "eye": eye, "target": target}
