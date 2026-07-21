"""Synthetic rotating star field with an optional moving trail.

Lets the whole pipeline be tested without a real capture: stars rotate rigidly
about a known pole at a known rate, and a bright straight streak crosses a few
frames to stand in for a satellite.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from remote_satellites.core.video import FrameWriter


def make_clip(path, *, n=60, w=640, h=480, center=(500.0, 360.0),
              total_deg=12.0, n_stars=45, trail=(25, 31), seed=0) -> dict:
    rng = np.random.default_rng(seed)
    cx, cy = center
    stars = rng.uniform([40, 40], [w - 40, h - 40], size=(n_stars, 2))
    bright = rng.uniform(140, 255, n_stars)

    with FrameWriter(path, w, h, 24, crf=6) as fw:
        for k in range(n):
            img = rng.normal(0, 2.0, (h, w, 3)).clip(0, 255).astype(np.uint8)
            ang = math.radians(total_deg) * (k / (n - 1))
            c, s = math.cos(ang), math.sin(ang)
            for (x0, y0), b in zip(stars, bright):
                dx, dy = x0 - cx, y0 - cy
                x = cx + c * dx - s * dy
                y = cy + s * dx + c * dy
                cv2.circle(img, (round(x), round(y)), 2,
                           (int(b), int(b), int(b)), -1)
            if trail and trail[0] <= k < trail[1]:
                # a bright streak marching left->right across the frame
                t = (k - trail[0]) / max(1, trail[1] - trail[0] - 1)
                x = int(60 + t * (w - 120))
                cv2.line(img, (x, 40), (x + 40, h - 40), (255, 255, 255), 2)
            fw.write(img)

    return {"center": center, "total_deg": total_deg, "n": n,
            "trail": trail, "w": w, "h": h}
