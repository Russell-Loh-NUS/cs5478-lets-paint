from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .paint_wall import Opening

WALL_NAMES = ("south", "east", "north", "west")

@dataclass(frozen=True)
class RoomConfig:
    # Default that we set for 4x3x2.6m room
    width: float = 4.0
    depth: float = 3.0
    height: float = 2.6
    thickness: float = 0.1
    openings: dict = field(default_factory=dict)

    @staticmethod
    def default() -> "RoomConfig":
        # Default obstacles - window at south, door east, window north
        return RoomConfig(openings={
            "south": (Opening(2.7, 0.9, 0.9, 1.0),),
            "east": (Opening(0.4, 0.0, 0.9, 2.05),),
            "north": (Opening(1.4, 0.9, 1.2, 1.0),),
        })

    def wall_length(self, name):
        return self.width if name in ("south", "north") else self.depth

    def validate(self):
        # Raise ValueError if an opening leaves its wall or two openings on a wall overlap.
        for name, ops in self.openings.items():
            if name not in WALL_NAMES:
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
        width, depth, height = u(self.width), u(self.depth), u(self.height)
        doors_left = self.max_doors
        openings = {}
        for name in WALL_NAMES:
            length = width if name in ("south", "north") else depth
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
        room = RoomConfig(width=width, depth=depth, height=height, openings=openings)
        room.validate()
        return room
