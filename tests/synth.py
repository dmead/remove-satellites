"""Synthetic rotating star field with an optional moving trail.

Lets the whole pipeline be tested without a real capture: stars rotate rigidly
about a known pole at a known rate, and a bright straight streak crosses a few
frames to stand in for a satellite.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from remove_satellites.core.video import FrameWriter


def make_clip(path, *, n=60, w=640, h=480, center=(500.0, 360.0),
              total_deg=12.0, n_stars=45, trail=(25, 31), seed=0,
              drift=(0.0, 0.0), ground=0, lights=0) -> dict:
    """drift: per-frame (dx, dy) sky translation, i.e. a pole so far
    off-frame the motion is a pan. ground: height of a static textured band
    along the bottom that occludes the sky like land. lights: bright
    camera-fixed points (floodlit structures, lamps), brighter and more
    numerous than the stars if you like."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    # scatter stars upstream of the drift so the field stays populated
    span = np.array(drift) * n
    lo = np.minimum([40, 40], np.array([40, 40]) - span)
    hi = np.maximum([w - 40, h - 40], np.array([w - 40, h - 40]) - span)
    n_stars = int(round(n_stars * np.prod(hi - lo) / ((w - 80) * (h - 80))))
    stars = rng.uniform(lo, hi, size=(n_stars, 2))
    bright = rng.uniform(140, 255, n_stars)

    lamps = rng.uniform([10, 10], [w - 10, h - 10], size=(lights, 2))
    lamp_b = rng.uniform(200, 255, lights)
    land = np.zeros((ground, w, 3), np.uint8)
    if ground:
        land[:] = 30
        for _ in range(ground * w // 150):          # rocks / shrubs
            x, y = rng.integers(0, w), rng.integers(0, ground)
            v = int(rng.integers(60, 200))
            cv2.circle(land, (int(x), int(y)), int(rng.integers(2, 6)),
                       (v, v, v), -1)

    with FrameWriter(path, w, h, 24, crf=6) as fw:
        for k in range(n):
            img = rng.normal(0, 2.0, (h, w, 3)).clip(0, 255).astype(np.uint8)
            ang = math.radians(total_deg) * (k / (n - 1))
            c, s = math.cos(ang), math.sin(ang)
            for (x0, y0), b in zip(stars, bright):
                dx, dy = x0 - cx, y0 - cy
                x = cx + c * dx - s * dy + drift[0] * k
                y = cy + s * dx + c * dy + drift[1] * k
                cv2.circle(img, (round(x), round(y)), 2,
                           (int(b), int(b), int(b)), -1)
            for (x, y), b in zip(lamps, lamp_b):
                cv2.circle(img, (round(x), round(y)), 3,
                           (int(b), int(b), int(b)), -1)
            if ground:
                img[h - ground:] = land
            if trail and trail[0] <= k < trail[1]:
                # a bright streak marching left->right across the frame
                t = (k - trail[0]) / max(1, trail[1] - trail[0] - 1)
                x = int(60 + t * (w - 120))
                cv2.line(img, (x, 40), (x + 40, h - 40), (255, 255, 255), 2)
            fw.write(img)

    return {"center": center, "total_deg": total_deg, "n": n,
            "trail": trail, "w": w, "h": h, "ground": ground}
