from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from .paint_canvas import PaintCanvas

OVERLAY_OFFSET = 5e-4 # 0.5mm off wall for overlay, should be offset in the -y direction of the actual wall to be in the interior


@dataclass(frozen=True)
class Opening:
    u: float
    v: float
    width: float
    height: float


@dataclass
class Wall:
    name: str
    origin: np.ndarray # world position of the face's u=0, v=0 corner
    u_dir: np.ndarray # unit vector along the face (horizontal)
    normal: np.ndarray # unit vector into the room (horizontal)
    length: float
    height: float
    thickness: float = 0.1
    openings: tuple = ()
    ext: tuple = (0.0, 0.0) # collision-only extension past u=0 / u=length (fills corners)
    canvas: PaintCanvas | None = None
    body_id: int = -1
    overlay_id: int = -1

    def __post_init__(self):
        for f in ("origin", "u_dir", "normal"):
            setattr(self, f, np.asarray(getattr(self, f), float))

    @property
    def yaw(self):
        return math.atan2(self.u_dir[1], self.u_dir[0])

    def uv_to_world(self, u, v):
        return self.origin + self.u_dir * u + np.array([0, 0, v])

    def world_to_uv(self, pt):
        d = np.asarray(pt, float) - self.origin
        return float(d @ self.u_dir), float(d[2])

    def signed_distance(self, pt):
        return float((np.asarray(pt, float) - self.origin) @ self.normal)

    def _pieces(self): # Pieces for simulating window, doors
        cuts = sorted({0.0, self.length, *[o.u for o in self.openings],
                       *[o.u + o.width for o in self.openings]})
        rects = []
        for ua, ub in zip(cuts, cuts[1:]):
            holes = sorted((o.v, o.v + o.height) for o in self.openings
                           if o.u <= ua + 1e-9 and o.u + o.width >= ub - 1e-9)
            v = 0.0
            for h0, h1 in holes:
                if h0 > v + 1e-9:
                    rects.append((ua, ub, v, h0))
                v = h1
            if v < self.height - 1e-9:
                rects.append((ua, ub, v, self.height))
        e0, e1 = self.ext
        if e0 > 0:
            rects.append((-e0, 0.0, 0.0, self.height))
        if e1 > 0:
            rects.append((self.length, self.length + e1, 0.0, self.height))
        return rects

    def spawn(self, work_dir, px_per_m, base_rgb, paint_rgb, wall_rgba):
        self.canvas = PaintCanvas(self.length, self.height, px_per_m, base_rgb, paint_rgb,
                                  self.openings)

        # Create wall
        pieces = self._pieces()
        t = self.thickness
        half = [[(u1 - u0) / 2, t / 2, (v1 - v0) / 2] for u0, u1, v0, v1 in pieces]
        pos = [[(u0 + u1) / 2, -t / 2, (v0 + v1) / 2] for u0, u1, v0, v1 in pieces]
        n = len(pieces)
        col = p.createCollisionShapeArray([p.GEOM_BOX] * n, halfExtents=half,
                                          collisionFramePositions=pos)
        vis = p.createVisualShapeArray([p.GEOM_BOX] * n, halfExtents=half,
                                       visualFramePositions=pos, rgbaColors=[wall_rgba] * n)
        orn = p.getQuaternionFromEuler([0, 0, self.yaw])
        self.body_id = p.createMultiBody(0, col, vis, basePosition=self.origin.tolist(),
                                         baseOrientation=orn)

        # Create overlay
        obj = os.path.join(work_dir, f"{self.name}_overlay.obj")
        self._write_overlay_obj(obj, [r for r in pieces if r[0] >= 0 and r[1] <= self.length])
        ovis = p.createVisualShape(p.GEOM_MESH, fileName=obj, rgbaColor=[1, 1, 1, 1])
        self.overlay_id = p.createMultiBody(0, -1, ovis, basePosition=self.origin.tolist(),
                                            baseOrientation=orn)
        self._work_dir = work_dir
        self.canvas.dirty = True
        self.push_texture()

    def _write_overlay_obj(self, path, rects): # We have an overlay for each wall face to simulate the 'paint'
        L, H, y = self.length, self.height, OVERLAY_OFFSET
        lines = ["vn 0 1 0"]
        idx = 0
        faces = []
        for u0, u1, v0, v1 in rects:
            corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # a b c d
            for u, v in corners:
                lines.append(f"v {u:.6f} {y:.6f} {v:.6f}")
                lines.append(f"vt {1 - u / L:.6f} {v / H:.6f}")
            a, b, c, d = (idx + i + 1 for i in range(4))
            faces += [f"f {a}/{a}/1 {d}/{d}/1 {c}/{c}/1", f"f {a}/{a}/1 {c}/{c}/1 {b}/{b}/1"]
            idx += 4
        with open(path, "w") as f:
            f.write("\n".join(lines + faces) + "\n")

    def push_texture(self): # This is needed so that textures can be refreshed from the robot's camera.
        c = self.canvas
        if not c.dirty:
            return
        from PIL import Image
        # Assign a new name each time we update the texture to prevent caching
        old, self._tex_version = getattr(self, "_tex_png", None), getattr(self, "_tex_version", -1) + 1
        self._tex_png = os.path.join(self._work_dir, f"{self.name}_tex{self._tex_version}.png")
        Image.fromarray(c.texture()).save(self._tex_png, compress_level=1)
        self.texture_id = p.loadTexture(self._tex_png)
        if old:
            os.remove(old)
        p.changeVisualShape(self.overlay_id, -1, textureUniqueId=self.texture_id)
        c.dirty = False

def build_room(work_dir, width=4.0, depth=3.0, height=2.6, thickness=0.1, px_per_m=250,
               base_rgb=(235, 232, 222), paint_rgb=(40, 90, 200), wall_rgba=(0.92, 0.91, 0.87, 1)):
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    t = thickness
    specs = [  # (name, origin, u_dir, normal, length, openings, ext)
        ("south", (0, 0, 0), x, y, width, (Opening(2.7, 0.9, 0.9, 1.0),), (t, t)),
        ("east", (width, 0, 0), y, -x, depth, (Opening(0.4, 0.0, 0.9, 2.05),), (0, 0)),
        ("north", (width, depth, 0), -x, -y, width, (Opening(1.4, 0.9, 1.2, 1.0),), (t, t)),
        ("west", (0, depth, 0), -y, x, depth, (), (0, 0)),
    ]
    walls = []
    for name, origin, u, n, length, openings, ext in specs:
        w = Wall(name, origin, u, n, length, height, t, openings, ext)
        w.spawn(work_dir, px_per_m, base_rgb, paint_rgb, wall_rgba)
        walls.append(w)
    return walls
