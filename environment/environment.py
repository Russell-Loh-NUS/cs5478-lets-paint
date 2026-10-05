from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data

from robot.camera import CameraConfig, PerceptionCamera
from robot.roller import PaintRoller, RollerConfig
from .wall import build_room


@dataclass
class PaintConfig:
    patch_width: float = 0.004 # width of the contact line perpendicular to the axis
    px_per_m: int = 250 # overlay canvas resolution
    gui_texture_every: int = 120 # steps between texture refreshes (2 Hz)
    paint_rgb: tuple = (40, 90, 200) # paint color


class PaintEnvironment:
    # Camera view
    CAMERA_VIEWS = { # View number corresponds to number on keyboard - each value is (distance, yaw, pitch, target)
        1: (2.8, 180, -8, (2.0, 0.0, 1.3)), # Face the south wall
        2: (3.5, 270, -8, (4.0, 1.5, 1.3)), # Face the east wall - door
        3: (2.8, 0, -8, (2.0, 3.0, 1.3)), # Face the north wall - window
        4: (3.5, 90, -8, (0.0, 1.5, 1.3)), # Face the west wall
        5: (5.5, 165, -40, (2.0, 1.5, 0.5)), # Overview from above the north wall
    }
    ORBIT_RATE, PAN_RATE, ZOOM_RATE = 90.0, 1.5, 1.5  # Deg/s, Meter/s, Factor/s

    def __init__(self, gui=False, paint=PaintConfig(), roller=RollerConfig(),
                 camera=CameraConfig(), dt=1 / 240):
        self.paint, self.dt, self.gui = paint, dt, gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self._cam_view = self.CAMERA_VIEWS[5]
            self._apply_cam()
            self._cam_keys_active = False
            print("Use your arrow keys to look around, number keys 1-5 for different views")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        self._work = tempfile.TemporaryDirectory(prefix="lets_paint_")
        # ENVIRONMENT INITIALIZATION
        # Floor
        floor = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 2.5, 0.05])
        floor_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 2.5, 0.05],
                                      rgbaColor=[0.55, 0.42, 0.3, 1])
        p.createMultiBody(0, floor, floor_v, basePosition=[2.0, 1.5, -0.05])
        # Paintable Walls
        self.walls = build_room(self._work.name, px_per_m=paint.px_per_m,
                                paint_rgb=paint.paint_rgb)
        self.roller = PaintRoller(roller)
        self.camera = PerceptionCamera(self.roller, camera, before_capture=self.flush_textures)
        self._steps = 0

    def step(self):
        p.stepSimulation()
        self._paint()
        self._steps += 1
        if self.gui and self._steps % self.paint.gui_texture_every == 0:
            self.flush_textures()
        if self.gui:
            self._camera_keys()
            time.sleep(self.dt)

    def _apply_cam(self):
        dist, yaw, pitch, target = self._cam_view
        p.resetDebugVisualizerCamera(dist, yaw, pitch, list(target))

    def _camera_keys(self): # FOR CAMERA MOVEMENT WITHIN SCENE
        keys = p.getKeyboardEvents()
        down = lambda k: keys.get(k, 0) & p.KEY_IS_DOWN
        for n in self.CAMERA_VIEWS:
            if keys.get(ord(str(n)), 0) & p.KEY_WAS_TRIGGERED:
                self._cam_view = self.CAMERA_VIEWS[n]
                self._apply_cam()
                self._cam_keys_active = False
                return
        dx = down(p.B3G_RIGHT_ARROW) - down(p.B3G_LEFT_ARROW)
        dy = down(p.B3G_UP_ARROW) - down(p.B3G_DOWN_ARROW)
        zoom = down(ord("=")) + down(ord("+")) - down(ord("-"))
        if not (dx or dy or zoom):
            self._cam_keys_active = False
            return
        if not self._cam_keys_active:
            _, _, _, _, _, _, _, _, yaw, pitch, dist, tgt = p.getDebugVisualizerCamera()
            self._cam_view = (dist, yaw, pitch, tuple(tgt))
            self._cam_keys_active = True
        dist, yaw, pitch, tgt = self._cam_view
        if down(p.B3G_SHIFT):  # pan the look-at point in the ground plane
            fwd = np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])
            right = np.array([fwd[1], -fwd[0]])
            move = (right * dx + fwd * dy) * self.PAN_RATE * self.dt
            tgt = (tgt[0] + move[0], tgt[1] + move[1], tgt[2])
        else:
            yaw += dx * self.ORBIT_RATE * self.dt
            pitch = float(np.clip(pitch + dy * self.ORBIT_RATE * self.dt, -89, 89))
        dist = float(np.clip(dist * self.ZOOM_RATE ** (-zoom * self.dt), 0.3, 20))
        self._cam_view = (dist, yaw, pitch, tgt)
        self._apply_cam()

    def flush_textures(self):
        for w in self.walls:
            w.push_texture()

    def _paint(self):
        a, b = self.roller.axis_segment() # Get the two ends of the roller
        for w in self.walls:
            contacts = p.getContactPoints(self.roller.id, w.body_id)
            if not contacts:
                w.canvas.lift()
                continue
            # Roller still in contact, continue stroke
            w.canvas.stroke(w.world_to_uv(a), w.world_to_uv(b), self.paint.patch_width)

    def painted_area(self):
        """Ground-truth painted area in m^2: ({wall name: area}, total)."""
        per_wall = {w.name: w.canvas.painted_area for w in self.walls}
        return per_wall, sum(per_wall.values())

    def close(self):
        p.disconnect(self.client)
        self._work.cleanup()
