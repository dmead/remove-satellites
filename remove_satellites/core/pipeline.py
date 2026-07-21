"""The cleaning pipeline: derotate → temporal median → re-rotate.

All frames are warped onto a padded canvas whose centre is the pole, so the
rotation loses no corner content. With the stars held still by derotation, a
sliding temporal median rejects transient streaks (satellites, planes) while
leaving the stars untouched; each median frame is then re-rotated back to its
original orientation and cropped to the source size.
"""

from __future__ import annotations

import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import stars, video
from .rotation import RotationModel

_NTHREADS = min(os.cpu_count() or 4, 16)


def _rot_matrix(center, angle_deg, tx=0.0, ty=0.0):
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                float(angle_deg), 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return m


def _auto_pad(model: RotationModel, w: int, h: int, ref_index: int) -> int:
    cx, cy = model.center
    amax = max(abs(model.omega * (0 - ref_index)),
               abs(model.omega * (model.n_frames - 1 - ref_index)))
    rmax = max(math.hypot(x - cx, y - cy)
               for x, y in [(0, 0), (w, 0), (0, h), (w, h)])
    return int(rmax * abs(math.sin(amax))) + 8


def _calibrate_sign(path, model: RotationModel, ref_index: int) -> float:
    """Return +1/-1 so that derotation actually undoes the measured rotation.

    Sign-convention insurance: rotate the farthest frame's star points both
    ways and keep whichever better matches the reference frame's stars.
    """
    n = model.n_frames
    k = 0 if ref_index > n // 2 else n - 1
    ref_xy = stars.detect(video.read_frame(path, ref_index))[:, :2]
    tgt_xy = stars.detect(video.read_frame(path, k))[:, :2]
    if len(ref_xy) < 4 or len(tgt_xy) < 4:
        return 1.0
    theta_deg = math.degrees(model.omega * (k - ref_index))
    tree = cKDTree(ref_xy)
    best_s, best_r = 1.0, math.inf
    for s in (1.0, -1.0):
        m = _rot_matrix(model.center, s * -theta_deg)
        rot = (m[:, :2] @ tgt_xy.T).T + m[:, 2]
        d, _ = tree.query(rot, distance_upper_bound=40.0)
        d = d[np.isfinite(d)]
        r = float(np.median(d)) if d.size else math.inf
        if r < best_r:
            best_s, best_r = s, r
    return best_s


def _median_u8(buf: list[np.ndarray], executor: ThreadPoolExecutor | None = None
               ) -> np.ndarray:
    """Per-pixel median across the window.

    The frame-axis median is the pipeline's hot loop, so we split the rows
    across threads (numpy's partition releases the GIL) and take the middle
    rank directly — for trail rejection the mid rank is as good as the true
    (even-window averaged) median, and cheaper.
    """
    stack = np.stack(buf, axis=0)                      # (k, H, W, 3)
    k = stack.shape[0]
    mid = k // 2
    if executor is None or k < 3:
        return np.partition(stack, mid, axis=0)[mid]

    rows = np.array_split(np.arange(stack.shape[1]), _NTHREADS)
    slices = [slice(r[0], r[-1] + 1) for r in rows if r.size]

    def strip(sl):
        return np.partition(stack[:, sl], mid, axis=0)[mid]

    return np.concatenate(list(executor.map(strip, slices)), axis=0)


class Cleaner:
    """Reusable engine for both full-clip export and single-frame preview."""

    def __init__(self, path, model: RotationModel, *, radius: int = 8,
                 ref_index: int | None = None, pad: int | None = None):
        self.path = path
        self.info = video.probe(path)
        self.model = model
        self.radius = int(radius)
        self.ref_index = (self.info.frame_count // 2
                          if ref_index is None else int(ref_index))
        self.pad = _auto_pad(model, self.info.width, self.info.height,
                             self.ref_index) if pad is None else int(pad)
        self.sign = _calibrate_sign(path, model, self.ref_index)
        self._cw = self.info.width + 2 * self.pad
        self._ch = self.info.height + 2 * self.pad

    def _angle_deg(self, k: int) -> float:
        # align to the clip's middle (self.ref_index) so the max derotation —
        # and thus the canvas padding — is half of the full sweep
        return math.degrees(self.model.omega * (k - self.ref_index))

    def _derotate(self, frame, k) -> np.ndarray:
        ang = self.sign * -self._angle_deg(k)
        m = _rot_matrix(self.model.center, ang, self.pad, self.pad)
        return cv2.warpAffine(frame, m, (self._cw, self._ch),
                              flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def _rerotate_crop(self, canvas, k) -> np.ndarray:
        ang = self.sign * self._angle_deg(k)
        pc = (self.model.center[0] + self.pad, self.model.center[1] + self.pad)
        m = _rot_matrix(pc, ang)
        out = cv2.warpAffine(canvas, m, (self._cw, self._ch),
                             flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        p = self.pad
        return out[p:p + self.info.height, p:p + self.info.width]

    def frame(self, index: int) -> np.ndarray:
        """Cleaned single frame (processes just its median window)."""
        r = self.radius
        lo = max(0, index - r)
        hi = min(self.info.frame_count - 1, index + r)
        buf = [self._derotate(video.read_frame(self.path, k), k)
               for k in range(lo, hi + 1)]
        with ThreadPoolExecutor(_NTHREADS) as ex:
            med = _median_u8(buf, ex)
        return self._rerotate_crop(med, index)

    def run(self, out_path, *, crf: int = 16, progress=None, cancel=None):
        """Stream the whole clip to `out_path` (audio copied from source).

        A single forward pass keeps a ring buffer of the last 2r+1 derotated
        frames. Output frames are emitted in strict order in three phases:
        leading (partial windows), centred (full windows), trailing (partial).
        """
        n = self.info.frame_count
        r = self.radius
        w_out = self.info.width
        h_out = self.info.height
        buf: deque[np.ndarray] = deque(maxlen=2 * r + 1)
        idx: deque[int] = deque(maxlen=2 * r + 1)
        state = {"written": 0}

        with video.FrameWriter(out_path, w_out, h_out, self.info.fps,
                               source=self.path, crf=crf) as fw, \
                ThreadPoolExecutor(_NTHREADS) as ex:
            reader = video.iter_frames(self.path)

            def emit(center: int) -> bool:
                lo, hi = max(0, center - r), min(n - 1, center + r)
                sub = [f for k, f in zip(idx, buf) if lo <= k <= hi]
                fw.write(self._rerotate_crop(_median_u8(sub, ex), center))
                state["written"] += 1
                if progress:
                    progress(state["written"], n)
                return bool(cancel and cancel())

            prime = min(2 * r + 1, n)
            for i in range(prime):
                buf.append(self._derotate(next(reader), i))
                idx.append(i)

            if n <= 2 * r + 1:                       # short clip: no streaming
                for c in range(n):
                    if emit(c):
                        return state["written"]
                return state["written"]

            for c in range(0, r + 1):                # leading (centres 0..r)
                if emit(c):
                    return state["written"]

            for i in range(2 * r + 1, n):            # centred (centres r+1..n-1-r)
                buf.append(self._derotate(next(reader), i))
                idx.append(i)
                if emit(i - r):
                    return state["written"]

            for c in range(n - r, n):                # trailing (centres n-r..n-1)
                if emit(c):
                    return state["written"]

        return state["written"]
