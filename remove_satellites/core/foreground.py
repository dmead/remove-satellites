"""Foreground (land, trees) mask: pixels fixed to the camera, not the sky.

The cleaner aligns each median window to the stars, which smears anything
fixed to the tripod. We find those pixels by asking, for frame pairs a few
dozen pixels of sky motion apart, which difference is smaller at each pixel:
the raw one (static content matches) or the sky-aligned one (stars match).
Frames are background-subtracted first so the camera-fixed sky glow doesn't
count as foreground.

The per-pixel vote is noisy and says nothing inside a uniform silhouette, so
it is turned into regions: close small gaps, then everything not reachable
from the top edge through sky pixels is foreground. Sky glimpsed through
branches ends up inside the tree's envelope — accepted for simplicity.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import stars, video
from .track import Track


def _prep(frame: np.ndarray) -> np.ndarray:
    g = stars.to_gray(frame).astype(np.float32)
    return cv2.GaussianBlur(g - stars.background(g), (3, 3), 0)


def static_mask(path, track: Track, *, pairs: int = 11,
                target_motion_px: float = 80.0, thresh: float = 3.0,
                close_px: int = 15) -> np.ndarray:
    """uint8 mask, 1 = foreground (camera-fixed), same size as the frames."""
    info = video.probe(path)
    n, w, h = info.frame_count, info.width, info.height

    good = np.flatnonzero(track.inliers > 0)   # skip washed-out stretches
    if good.size < 2:
        return np.zeros((h, w), np.uint8)
    motion = np.median([track.motion_px(int(k), w, h) for k in good[::10]])
    gap = int(np.clip(round(target_motion_px / max(motion, 1e-3)),
                      2, max(2, n // 4)))

    lo, hi = int(good[0]), int(good[-1]) + 1 - gap
    if hi <= lo:
        return np.zeros((h, w), np.uint8)
    votes = []
    for k in np.unique(np.linspace(lo, hi, pairs).astype(int)):
        a = _prep(video.read_frame(path, int(k)))
        b = _prep(video.read_frame(path, int(k) + gap))
        aw = cv2.warpPerspective(a, track.between(int(k), int(k) + gap),
                                 (w, h), borderMode=cv2.BORDER_REPLICATE)
        votes.append(np.abs(aw - b) - np.abs(a - b))
    score = cv2.GaussianBlur(np.median(np.stack(votes), axis=0), (0, 0), 2)

    static = (score > thresh).astype(np.uint8)
    if close_px > 1:
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(
                                      cv2.MORPH_ELLIPSE, (close_px, close_px)))
    sky = 1 - static
    _, lab = cv2.connectedComponents(sky, connectivity=4)
    seeds = np.unique(lab[0][sky[0] == 1])
    return (~np.isin(lab, seeds)).astype(np.uint8)
