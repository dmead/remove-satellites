"""The cleaning pipeline: per-window star alignment → temporal median.

For each output frame c, the 2r neighbours are warped onto c's own geometry
with the sky-motion homography between them (`Track.between`), so the stars
stand still across the window and a per-pixel median rejects the transient
streaks (satellites, planes) without smearing a star. Aligning each window to
its own centre means no global derotation, no canvas padding and no second
re-rotation interpolation, and it works for any pole position — including
far off-frame, where the sky motion is near-translation and lens projection
makes it anything but a rigid rotation.

Camera-fixed foreground (`foreground.static_mask`) would smear under star
alignment, so it gets a plain (unaligned) median instead, and aligned samples
that land on foreground or outside the neighbour's frame are excluded from
the sky median.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import shutil
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import hw, stars, video
from .rotation import RotationModel
from .track import Track

_NTHREADS = min(os.cpu_count() or 4, 32)


def _rot_matrix(center, angle_deg, tx=0.0, ty=0.0):
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                float(angle_deg), 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return m


def _calibrate_sign(path, model: RotationModel, ref_index: int) -> float:
    """Return +1/-1 so that the rotation model's sense matches the sky.

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
    """Reusable engine for both full-clip export and single-frame preview.

    `motion` is a `Track` (per-step homographies) or a `RotationModel` (pole +
    rate, converted to a uniform track). `fg_mask` (1 = camera-fixed
    foreground) is optional; without it every pixel is treated as sky.
    """

    def __init__(self, path, motion: Track | RotationModel, *,
                 radius: int = 8, fg_mask: np.ndarray | None = None):
        self.path = path
        self.info = video.probe(path)
        self.radius = int(radius)
        if isinstance(motion, RotationModel):
            sign = _calibrate_sign(path, motion, self.info.frame_count // 2)
            motion = Track.from_rotation(motion, sign)
        self.track = motion
        self.fg = None if fg_mask is None or not fg_mask.any() else \
            fg_mask.astype(np.uint8)
        if self.fg is not None:
            rows = np.flatnonzero(self.fg.any(axis=1))
            self._fg_rows = slice(int(rows[0]), int(rows[-1]) + 1)
            soft = cv2.GaussianBlur(self.fg.astype(np.float32), (0, 0), 2.0)
            self._fg_alpha = soft[self._fg_rows, :, None]

    def _aligned(self, frame: np.ndarray, src: int, dst: int) -> np.ndarray:
        """Frame `src` warped onto frame `dst`; excluded samples get 0/255.

        The median can't take per-pixel sample counts, so invalid samples are
        filled alternately with 0 and 255 by frame parity: equal numbers of
        each below and above leave the middle rank on the valid samples.
        """
        w, h = self.info.width, self.info.height
        H = self.track.between(src, dst)
        out = cv2.warpPerspective(frame, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        src_ok = np.ones((h, w), np.uint8) if self.fg is None else 1 - self.fg
        ok = cv2.warpPerspective(src_ok, H, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[ok == 0] = 0 if (src - dst) % 2 == 0 else 255
        return out

    def _combine(self, center: int, raw: dict[int, np.ndarray],
                 ex: ThreadPoolExecutor) -> np.ndarray:
        """Median of the window around `center` from its raw frames."""
        keys = sorted(raw)
        warped = list(ex.map(
            lambda k: raw[k] if k == center else
            self._aligned(raw[k], k, center), keys))
        sky = _median_u8(warped, ex)
        if self.fg is None:
            return sky
        rs = self._fg_rows
        plain = _median_u8([raw[k][rs] for k in keys], ex)
        a = self._fg_alpha
        sky[rs] = (a * plain + (1.0 - a) * sky[rs] + 0.5).astype(np.uint8)
        return sky

    def frame(self, index: int, cache: video.FrameCache | None = None
              ) -> np.ndarray:
        """Cleaned single frame (processes just its median window).

        The window is decoded in one sequential pass from a single seek (not
        2r+1 random seeks); with a `cache`, frames decoded for earlier
        previews are reused, so scrubbing mostly skips decoding."""
        r = self.radius
        lo = max(0, index - r)
        hi = min(self.info.frame_count - 1, index + r)
        raw = {}
        for k in range(lo, hi + 1):
            f = cache.get(k) if cache is not None else None
            if f is not None:
                raw[k] = f
        missing = [k for k in range(lo, hi + 1) if k not in raw]
        if missing:
            for k, f in enumerate(video.iter_frames(
                    self.path, missing[0], missing[-1] + 1), missing[0]):
                if k not in raw:
                    raw[k] = f
                    if cache is not None:
                        cache.put(k, f)
        with ThreadPoolExecutor(_NTHREADS) as ex:
            return self._combine(index, raw, ex)

    def _render(self, lo: int, hi: int, emit, *, threads: int,
                progress=None, cancel=None) -> bool:
        """Emit cleaned frames lo..hi-1 in order. One sequential read of
        lo-r .. hi-1+r through a ring buffer holding the current window.
        Returns False if cancelled."""
        n, r = self.info.frame_count, self.radius
        buf: deque[tuple[int, np.ndarray]] = deque(maxlen=2 * r + 1)
        start, stop = max(0, lo - r), min(n, hi + r)
        reader = enumerate(video.iter_frames(self.path, start, stop), start)
        with ThreadPoolExecutor(threads) as ex:
            for c in range(lo, hi):
                top = min(n - 1, c + r)
                while not buf or buf[-1][0] < top:
                    buf.append(next(reader))
                emit(self._combine(c, {k: f for k, f in buf
                                       if c - r <= k <= top}, ex))
                if progress:
                    progress(1)
                if cancel and cancel():
                    return False
        return True

    def run(self, out_path, *, crf: int = 16, progress=None, cancel=None,
            workers: int | None = None):
        """Render the whole clip to `out_path` (audio copied from source).

        One output frame costs ~1.2 s at 4K and threads stop helping past ~8
        (memory bandwidth, serial copies), so the clip is split into
        contiguous chunks rendered by several processes, each encoding its
        own segment; the segments are then joined losslessly and the audio
        muxed. How many processes and threads: `hw.plan` (or `workers`).
        Returns frames written.
        """
        n = self.info.frame_count
        p = hw.plan(self.info.width, self.info.height, self.radius,
                    workers=workers)
        w = min(p.export_workers, max(1, n // (4 * self.radius + 2)))
        done = 0

        def tick(k=1):
            nonlocal done
            done += k
            if progress:
                progress(done, n)

        if w == 1:
            with video.FrameWriter(out_path, self.info.width,
                                   self.info.height, self.info.fps,
                                   source=self.path, crf=crf) as fw:
                self._render(0, n, fw.write, threads=_NTHREADS,
                             progress=tick, cancel=cancel)
            return done

        out_path = Path(out_path)
        tmp = Path(tempfile.mkdtemp(prefix=".rs-segments-",
                                    dir=out_path.parent))
        bounds = np.linspace(0, n, w + 1).astype(int)
        segs = [tmp / f"seg{i:03d}.mp4" for i in range(w)]
        threads = max(1, p.cores // w)
        spec = (str(self.path), self.track, self.radius, self.fg)
        try:
            with multiprocessing.Manager() as mgr, \
                    ProcessPoolExecutor(w) as ex:
                q, stop = mgr.Queue(), mgr.Event()
                futs = [ex.submit(_render_segment, spec, int(a), int(b),
                                  str(s), crf, threads, q, stop)
                        for a, b, s in zip(bounds[:-1], bounds[1:], segs)]
                pending = set(futs)
                while pending:
                    _, pending = wait(pending, timeout=0.2)
                    while not q.empty():
                        tick(q.get())
                    if cancel and cancel():
                        stop.set()
                for f in futs:
                    f.result()                      # re-raise worker errors
                while not q.empty():
                    tick(q.get())
                if stop.is_set():
                    return done
            video.concat(segs, out_path, source=self.path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return done


def _render_segment(spec, lo, hi, seg_path, crf, threads, q, stop) -> bool:
    """Worker process: render frames lo..hi-1 of the clip to one segment."""
    path, motion, radius, fg = spec
    c = Cleaner(path, motion, radius=radius, fg_mask=fg)
    global _NTHREADS
    _NTHREADS = threads                  # median strips: this process's share
    with video.FrameWriter(seg_path, c.info.width, c.info.height,
                           c.info.fps, crf=crf, threads=threads) as fw:
        return c._render(lo, hi, fw.write, threads=threads,
                         progress=q.put, cancel=stop.is_set)
