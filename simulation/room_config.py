from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .paint_wall import Opening

WALL_NAMES = ("south", "east", "north", "west")
NOTCH_CORNERS = ("sw", "se", "ne", "nw")
# Name of a wall by the direction its u axis runs; extra walls in an L room get a numeric suffix.
_DIR_NAMES = {(1, 0): "south", (0, 1): "east", (-1, 0): "north", (0, -1): "west"}

@dataclass(frozen=True)
class RoomConfig:
    # Default that we set for 4x3x2.6m room
    width: float = 4.0
    depth: float = 3.0
    height: float = 2.6
    thickness: float = 0.1
    openings: dict = field(default_factory=dict)
    # None for a rectangle. For an L shape, (corner, notch_width, notch_depth): the rectangle
    # notch_width (x) by notch_depth (y) is cut out of that corner ("sw", "se", "ne" or "nw")
    # of the width x depth bounding box.
    notch: tuple | None = None

    @staticmethod
    def default() -> "RoomConfig":
        # Default obstacles - window at south, door east, window north
        return RoomConfig(openings={
            "south": (Opening(2.7, 0.9, 0.9, 1.0),),
            "east": (Opening(0.4, 0.0, 0.9, 2.05),),
            "north": (Opening(1.4, 0.9, 1.2, 1.0),),
        })

    def footprint(self):
        # Corners of the floor plan, counter-clockwise, starting from the first corner at the south.
        w, d = self.width, self.depth
        if self.notch is None:
            return [(0, 0), (w, 0), (w, d), (0, d)]
        corner, a, b = self.notch
        return {
            "ne": [(0, 0), (w, 0), (w, d - b), (w - a, d - b), (w - a, d), (0, d)],
            "se": [(0, 0), (w - a, 0), (w - a, b), (w, b), (w, d), (0, d)],
            "nw": [(0, 0), (w, 0), (w, d), (a, d), (a, d - b), (0, d - b)],
            "sw": [(a, 0), (w, 0), (w, d), (0, d), (0, b), (a, b)],
        }[corner]

    def wall_specs(self):
        # One (name, origin, u_dir, normal, length, ext) per wall, following the footprint. The
        # normal points into the room. ext is the collision-only extension past u=0 / u=length
        # that fills the notch left outside a convex corner (reflex corners need none).
        pts = np.array(self.footprint(), float)
        n, t = len(pts), self.thickness
        edges = [(pts[(i + 1) % n] - pts[i]) for i in range(n)]
        seen, specs = {}, []
        for i, e in enumerate(edges):
            length = float(np.hypot(*e))
            u = e / length
            base = _DIR_NAMES[(int(round(u[0])), int(round(u[1])))]
            seen[base] = seen.get(base, 0) + 1
            name = base if seen[base] == 1 else f"{base}_{seen[base]}"
            prev = edges[i - 1] / np.hypot(*edges[i - 1])
            convex = prev[0] * u[1] - prev[1] * u[0] > 0
            specs.append((name, (pts[i][0], pts[i][1], 0.0), np.array([u[0], u[1], 0.0]),
                          np.array([-u[1], u[0], 0.0]), length, (t if convex else 0.0, 0.0)))
        return specs

    def wall_names(self):
        return [s[0] for s in self.wall_specs()]

    def wall_length(self, name):
        for spec in self.wall_specs():
            if spec[0] == name:
                return spec[4]
        raise ValueError(f"unknown wall {name!r}")

    def room_start_pos(self):
        # A point well inside the room (the bounding-box center can fall in an L's notch): the
        # center of the larger of the two rectangles an L splits into.
        w, d = self.width, self.depth
        if self.notch is None:
            return (w / 2, d / 2)
        corner, a, b = self.notch
        top, right = corner[0] == "n", corner[1] == "e"
        wide = ((0.0, w), (0.0, d - b) if top else (b, d)) # full width, rows without the notch
        tall = ((0.0, w - a) if right else (a, w), (0.0, d)) # full depth, columns without it
        (x0, x1), (y0, y1) = max(wide, tall, key=lambda r: (r[0][1] - r[0][0]) * (r[1][1] - r[1][0]))
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def validate(self):
        # Raise ValueError if the notch doesn't leave an L, an opening leaves its wall or two
        # openings on a wall overlap.
        if self.notch is not None:
            corner, a, b = self.notch
            if corner not in NOTCH_CORNERS:
                raise ValueError(f"unknown notch corner {corner!r}")
            if not (0 < a < self.width and 0 < b < self.depth):
                raise ValueError(f"notch {a} x {b} does not fit inside {self.width} x {self.depth}")
        names = self.wall_names()
        for name, ops in self.openings.items():
            if name not in names:
                raise ValueError(f"unknown wall {name!r}")
            length = self.wall_length(name)
            for o in ops:
                if o.u < 0 or o.u + o.width > length or o.v < 0 or o.v + o.height > self.height:
                    raise ValueError(f"{name} wall: opening {o} is outside the wall face")
            for i, a in enumerate(ops):
                for b in ops[i + 1:]:
                    if a.u < b.u + b.width and b.u < a.u + a.width:
                        raise ValueError(f"{name} wall: openings {a} and {b} overlap")


@dataclass
class RoomSampler:
    # Draws random RoomConfigs. Same seed always gives the same room.
    width: tuple = (3.0, 6.0)
    depth: tuple = (3.0, 5.0)
    height: tuple = (2.4, 3.0)
    l_shape_probability: float = 0.0 # chance a room is L shaped (0 = always a rectangle, 1 = always L)
    l_width: tuple = (4.0, 6.0) # bounding box of an L room
    l_depth: tuple = (4.0, 5.5)
    notch_size: tuple = (1.0, 3.0) # largest notch dimension is also capped by min_arm
    min_arm: float = 1.8 # narrowest the L may get, so the robot can fit through
    max_openings_per_wall: int = 2
    max_doors: int = 1 # per room
    door_probability: float = 0.25 # chance an opening is a door rather than a window
    door_width: tuple = (0.8, 1.0)
    door_height: tuple = (1.9, 2.1)
    window_width: tuple = (0.6, 1.6)
    window_height: tuple = (0.8, 1.4)
    window_sill: tuple = (0.7, 1.1) # window bottom edge above the floor
    edge_margin: float = 0.3 # min gap between an opening and a wall end / the next one
    top_margin: float = 0.2 # min gap between an opening and the ceiling

    def sample(self, seed=None) -> RoomConfig:
        rng = np.random.default_rng(seed)
        u = lambda r: float(rng.uniform(*r))
        # Only draws from rng when L rooms are enabled, so rectangular rooms keep their seeds.
        is_l = self.l_shape_probability > 0 and rng.random() < self.l_shape_probability
        width, depth = (u(self.l_width), u(self.l_depth)) if is_l else (u(self.width), u(self.depth))
        height = u(self.height)
        notch = None
        if is_l:
            corner = NOTCH_CORNERS[int(rng.integers(len(NOTCH_CORNERS)))]
            a = float(rng.uniform(self.notch_size[0], max(self.notch_size[0], min(self.notch_size[1], width - self.min_arm))))
            b = float(rng.uniform(self.notch_size[0], max(self.notch_size[0], min(self.notch_size[1], depth - self.min_arm))))
            notch = (corner, a, b)
        names_lengths = [(s[0], s[4]) for s in RoomConfig(width=width, depth=depth, notch=notch).wall_specs()]
        doors_left = self.max_doors
        openings = {}
        for name, length in names_lengths:
            placed = []
            for _ in range(int(rng.integers(0, self.max_openings_per_wall + 1))):
                is_door = doors_left > 0 and rng.random() < self.door_probability
                if is_door:
                    w, h, v = u(self.door_width), u(self.door_height), 0.0
                else:
                    w, h, v = u(self.window_width), u(self.window_height), u(self.window_sill)
                h = min(h, height - self.top_margin - v)
                if h <= 0.3 or length - 2 * self.edge_margin < w:
                    continue
                for _ in range(20): # random placement that keeps clear of the other openings
                    o = Opening(float(rng.uniform(self.edge_margin, length - w - self.edge_margin)), v, w, h)
                    if all(o.u >= q.u + q.width + self.edge_margin or q.u >= o.u + o.width + self.edge_margin
                           for q in placed):
                        placed.append(o)
                        doors_left -= is_door
                        break
            if placed:
                openings[name] = tuple(sorted(placed, key=lambda o: o.u))
        room = RoomConfig(width=width, depth=depth, height=height, openings=openings, notch=notch)
        room.validate()
        return room
