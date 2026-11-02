from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from .paint_wall import build_room
from .room_config import RoomConfig, RoomSampler


@dataclass
class PaintConfig:
    patch_width: float = 0.004 # width of the contact line perpendicular to the axis
    px_per_m: int = 250 # overlay canvas resolution
    gui_texture_every: int = 120 # steps between texture refreshes (2 Hz)
    paint_rgb: tuple = (40, 90, 200) # paint color
    render_paint: bool = True # if false keep only the paint mask (no overlay/textures)
    end_tolerance: float = 0.01 # m, each roller end may be this far beyond one radius from the wall


class PaintEnvironment:
    # Maps a wall name to its matching numbered view above, for auto-following the active wall.
    WALL_VIEW_NUMBER = {"south": 1, "east": 2, "north": 3, "west": 4}
    ORBIT_RATE, PAN_RATE, ZOOM_RATE = 90.0, 1.5, 1.5  # Deg/s, Meter/s, Factor/s

    def __init__(self, gui=False, paint=PaintConfig(), dt=1 / 240, room=None, seed=None,
                 sampler=RoomSampler(), video_path=None):
        # If room is given, we use it. Otherwise a random one drawn using sampler based on the seed
        # else, use default
        self.paint, self.dt, self.gui, self.sampler = paint, dt, gui, sampler
        self.video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
        self._logging_id = None
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self._cam_keys_active = False
            if self.video_path is not None:
                self.video_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, str(self.video_path))
                    print(f"[PaintEnvironment] Recording video to {self.video_path}")
                except Exception as exc:
                    print(f"[PaintEnvironment] Video recording unavailable: {exc}")
            print("Use your arrow keys to look around, number keys 1-5 for different views")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        self._work = tempfile.TemporaryDirectory(prefix="paint_env_")
        self.floor_id = None
        self.walls = []
        self._build_room(room, seed)

    def _build_room(self, room, seed):
        # Recreate the floor, walls and camera presets for a room layout
        if room is None:
            room = self.sampler.sample(seed) if seed is not None else RoomConfig.default()
        self.room, self.seed = room, seed
        w, d, h = room.width, room.depth, room.height
        margin = 1.0  # floor extends this far past the walls
        half = [w / 2 + margin, d / 2 + margin, 0.05]
        self.floor_id = p.createMultiBody(
            0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
            p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.42, 0.3, 1]),
            basePosition=[w / 2, d / 2, -0.05])
        self.walls = build_room(self._work.name, room, px_per_m=self.paint.px_per_m,
                                paint_rgb=self.paint.paint_rgb,
                                render_paint=self.paint.render_paint)
        self.walls_by_name = {wall.name: wall for wall in self.walls}
        self._steps = 0
        # Camera presets as (distance, yaw, pitch, target); the key is the number key that picks it.
        self.camera_views = {
            1: (d - 0.2, 180, -8, (w / 2, 0.0, h / 2)), # face the south wall
            2: (w - 0.5, 270, -8, (w, d / 2, h / 2)), # face the east wall
            3: (d - 0.2, 0, -8, (w / 2, d, h / 2)), # face the north wall
            4: (w - 0.5, 90, -8, (0.0, d / 2, h / 2)), # face the west wall
            5: (max(w, d) + 1.5, 165, -40, (w / 2, d / 2, 0.5)), # overview from above the north wall
        }
        if self.gui:
            self._cam_view = self.camera_views[5]
            self._apply_cam()

    def reset(self, seed=None, room=None):
        # Replace the room with a new layout (a given `room`, else one sampled with `seed`,
        # else the default) and clear all paint. Bodies outside the room (robot, paint can) are
        # left alone, so the caller must put them back where they belong. Returns the RoomConfig.
        for wall in self.walls:
            wall.remove()
        if self.floor_id is not None:
            p.removeBody(self.floor_id)
        self._build_room(room, seed)
        return self.room

    def step(self):
        p.stepSimulation()
        self._steps += 1
        if self.gui and self._steps % self.paint.gui_texture_every == 0:
            self.flush_textures()
        if self.gui:
            self._camera_keys()
            # Sleeping every single step paces the whole run to literal real-time (240 steps/s),
            # which made full routes take tens of minutes once the planning grid got finer (more
            # total moves). Sleeping only a fraction of steps still looks smooth -- PyBullet
            # renders every step regardless -- but lets bulk execution run several times faster.
            if self._steps % 4 == 0:
                time.sleep(self.dt)

    def _apply_cam(self):
        dist, yaw, pitch, target = self._cam_view
        p.resetDebugVisualizerCamera(dist, yaw, pitch, list(target))

    def focus_camera_on_wall(self, name):
        """Switch to the numbered camera view matching the given wall name (see camera_views)."""
        if not self.gui:
            return
        view_number = self.WALL_VIEW_NUMBER.get(name.split("_")[0]) # "east_2" -> the east view
        if view_number is None:
            return
        self._cam_view = self.camera_views[view_number]
        self._apply_cam()

    def _camera_keys(self): # FOR CAMERA MOVEMENT WITHIN SCENE
        keys = p.getKeyboardEvents()
        down = lambda k: keys.get(k, 0) & p.KEY_IS_DOWN
        for n in self.camera_views:
            if keys.get(ord(str(n)), 0) & p.KEY_WAS_TRIGGERED:
                self._cam_view = self.camera_views[n]
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

    def paint_contact(self, roller_body_id, axis_a, axis_b, roller_link_index=-1,
                      roller_radius=0.06):
        """Call with a roller body's two axis endpoints (world coords) every physics step it
        moves; paints a stroke onto whichever wall(s) the roller LINK specifically is touching
        (not any other part of a multi-link body), based on real contact detection.

        The roller only paints when BOTH of its ends are touching the wall (see
        _both_ends_touching), so a tilted roller touching with one end does not paint."""
        painted_any = False
        for w in self.walls:
            contacts = p.getContactPoints(roller_body_id, w.body_id, linkIndexA=roller_link_index)
            if not contacts or not self._both_ends_touching(w, axis_a, axis_b, roller_radius):
                w.canvas.lift()
                continue
            w.canvas.stroke(w.world_to_uv(axis_a), w.world_to_uv(axis_b), self.paint.patch_width)
            painted_any = True
        return painted_any

    def _both_ends_touching(self, wall, axis_a, axis_b, radius):
        """True if each end of the roller's axis is within one radius (plus `end_tolerance`) of
        the wall plane. Contact points can't be used to tell the ends apart: a roller lying flat
        often reports only a single one, so this is checked geometrically."""
        reach = radius + self.paint.end_tolerance
        return wall.signed_distance(axis_a) <= reach and wall.signed_distance(axis_b) <= reach

    def painted_area(self):
        """Ground-truth painted area in m^2: ({wall name: area}, total)."""
        per_wall = {w.name: w.canvas.painted_area for w in self.walls}
        return per_wall, sum(per_wall.values())

    def save_video(self):
        if not self.gui or self.video_path is None:
            return
        if self._logging_id is None:
            return
        try:
            p.stopStateLogging(self._logging_id)
            self._logging_id = None
            print(f"[PaintEnvironment] Saved video to {self.video_path}")
            if not self.video_path.exists():
                print(f"[PaintEnvironment] Warning: video file was not created at {self.video_path}")
        except Exception as exc:
            print(f"[PaintEnvironment] Failed to save final video: {exc}")

    def close(self):
        try:
            if self.gui and self.video_path is not None:
                self.save_video()
            if p.isConnected(self.client):
                p.disconnect(self.client)
        finally:
            self._work.cleanup()
