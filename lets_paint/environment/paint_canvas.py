from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw # We use pillow for drawing the 'paint' texture


class PaintCanvas:
    def __init__(self, length, height, px_per_m, base_rgb, paint_rgb, openings=()):
        self.length, self.height, self.ppm = length, height, px_per_m
        self.width_px = int(round(length * px_per_m))
        self.height_px = int(round(height * px_per_m))
        self.base_rgb = np.array(base_rgb, np.uint8)
        self.paint_rgb = np.array(paint_rgb, np.uint8)

        # Use pillow for the masking
        self._mask = Image.new("L", (self.width_px, self.height_px), 0)
        self._draw = ImageDraw.Draw(self._mask)
        self.paintable = np.ones((self.height_px, self.width_px), bool)
        for o in openings:
            c0, r0 = self.uv_to_px(o.u + o.width, o.v + o.height)
            c1, r1 = self.uv_to_px(o.u, o.v)
            self.paintable[int(round(r0)):int(round(r1)), int(round(c0)):int(round(c1))] = False

        self._prev = None
        self.dirty = True

    def uv_to_px(self, u, v):
        return (self.length - u) * self.ppm, (self.height - v) * self.ppm

    def stroke(self, a_uv, b_uv, width):
        # Draw on mask to simulate the 'paint'
        a, b = self.uv_to_px(*a_uv), self.uv_to_px(*b_uv)
        if self._prev is not None:
            pa, pb = self._prev
            self._draw.polygon([pa, pb, b, a], fill=255)
        self._draw.line([a, b], fill=255, width=max(1, round(width * self.ppm)))
        self._prev = (a, b)
        self.dirty = True

    def lift(self):
        self._prev = None

    @property
    def painted(self) -> np.ndarray:
        return (np.asarray(self._mask) > 0) & self.paintable

    @property
    def painted_area(self) -> float: # Painted area in m^2
        return float(self.painted.sum()) / self.ppm ** 2

    @property
    def paintable_area(self) -> float: # Paintable area in m^2
        return float(self.paintable.sum()) / self.ppm ** 2

    def texture(self) -> np.ndarray: # Texture is basically just returning the paint color on the drawn parts on mask
        m = np.asarray(self._mask) > 0
        return np.where(m[..., None], self.paint_rgb, self.base_rgb).astype(np.uint8)
