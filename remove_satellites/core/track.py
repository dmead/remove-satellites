"""Frame-to-frame sky motion as a chain of homographies.

The pole + rate model (`rotation.py`) assumes the sky turns as a rigid 2D
rotation in the image. That breaks when the pole is far off-frame (trails
nearly straight) and when a wide-angle lens is aimed low: the projected sky
motion then isn't a similarity at all — its apparent rotation and scale drift
across the clip. A homography per step captures any of it.

Only *local* alignment is needed: the temporal median aligns each window's
neighbours to its centre frame, so we store one homography per adjacent pair
and compose at most 2r of them. Chaining 10 steps (~50 px of star motion)
reproduces direct tracking to ~0.3 px.

Convention: `steps[k]` maps frame-k pixel coords to frame-(k+1) coords.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import stars, video
from .rotation import RotationModel


@dataclass(frozen=True)
class Track:
    steps: np.ndarray            # (n-1, 3, 3) float64, frame k -> k+1
    inliers: np.ndarray          # (n-1,) RANSAC inliers per step (0 = filled)
    residual_px: float           # median per-step star residual

    @property
    def n_frames(self) -> int:
        return len(self.steps) + 1

    def between(self, src: int, dst: int) -> np.ndarray:
        """Homography mapping frame `src` coords to frame `dst` coords."""
        h = np.eye(3)
        if dst >= src:
            for k in range(src, dst):
                h = self.steps[k] @ h
        else:
            for k in range(src - 1, dst - 1, -1):     # undo k-1 first
                h = np.linalg.inv(self.steps[k]) @ h
        return h / h[2, 2]

    def motion_px(self, k: int, w: int, h: int) -> float:
        """Median displacement (px) of a grid of points over step k."""
        xs, ys = np.meshgrid(np.linspace(0, w, 9), np.linspace(0, h, 5))
        p = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.steps[k])
        return float(np.median(np.hypot(*(q.reshape(-1, 2) - p).T)))

    def save(self, path) -> None:
        np.savez(path, steps=self.steps, inliers=self.inliers,
                 residual_px=self.residual_px)

    @classmethod
    def load(cls, path) -> "Track":
        z = np.load(path)
        return cls(steps=z["steps"], inliers=z["inliers"],
                   residual_px=float(z["residual_px"]))

    @classmethod
    def from_rotation(cls, model: RotationModel, sign: float = 1.0) -> "Track":
        """Express a pole + rate model as a (uniform) step chain."""
        m = cv2.getRotationMatrix2D(
            (float(model.center[0]), float(model.center[1])),
            sign * math.degrees(model.omega), 1.0)
        step = np.vstack([m, [0.0, 0.0, 1.0]])
        n = model.n_frames
        return cls(steps=np.repeat(step[None], n - 1, axis=0),
                   inliers=np.zeros(n - 1, int),
                   residual_px=model.residual_px)


def default_cache(path) -> Path:
    """Where a clip's track is cached: next to it, `<stem>.track.npz`."""
    p = Path(path)
    return p.with_name(p.stem + ".track.npz")


def load_or_estimate(path, cache=None, progress=None) -> tuple[Track, bool]:
    """(track, from_cache). Tracking 4K takes minutes, so the result is
    cached next to the clip and reused by both the CLI and the GUI."""
    cache = Path(cache) if cache else default_cache(path)
    if cache.exists():
        return Track.load(cache), True
    tr = estimate(path, progress=progress)
    try:
        tr.save(cache)
    except OSError:
        pass                                   # read-only source dir: fine
    return tr, False


def summary(tr: Track, w: int, h: int) -> str:
    mot = [tr.motion_px(k, w, h) for k in range(len(tr.steps))]
    filled = int((tr.inliers == 0).sum())
    return (f"{np.median(mot):.2f} px/frame star motion, residual "
            f"{tr.residual_px:.2f} px, {filled}/{len(tr.steps)} steps filled")


def _step(g0: np.ndarray, g1: np.ndarray, pts: np.ndarray,
          ransac_px: float, static_px: float = 0.3
          ) -> tuple[np.ndarray | None, int, float]:
    if len(pts) < 8:
        return None, 0, math.nan
    p = pts.astype(np.float32).reshape(-1, 1, 2)
    q, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p, None, winSize=(11, 11),
                                        maxLevel=3)
    ok = st.ravel() == 1
    # camera-fixed features (rocks, lit trees) can outnumber the stars and
    # win the RANSAC vote with an identity fit; the sky is what moves, so
    # fit to the movers when there are enough of them
    moving = ok & (np.hypot(*(q - p).reshape(-1, 2).T) > static_px)
    if moving.sum() >= max(8, ok.sum() // 10):
        ok = moving
    if ok.sum() < 8:
        return None, 0, math.nan
    h, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, ransac_px)
    if h is None:
        return None, 0, math.nan
    inl = inl.ravel().astype(bool)
    r = np.hypot(*(cv2.perspectiveTransform(p[ok][inl], h)
                   - q[ok][inl]).reshape(-1, 2).T)
    return h / h[2, 2], int(inl.sum()), float(np.median(r))


def _steps_chunk(path, a: int, b: int, max_stars: int, ransac_px: float,
                 min_inliers: int):
    """Raw fits for steps a..b-1 (needs frames a..b), read sequentially.

    Module-level so worker processes can import it."""
    m = b - a
    steps = np.full((m, 3, 3), np.nan)
    inl = np.zeros(m, int)
    res = np.full(m, np.nan)
    prev_g = prev_pts = None
    for i, frame in enumerate(video.iter_frames(path, a, b + 1)):
        g = stars.to_gray(frame)
        if prev_g is not None:
            h, ni, r = _step(prev_g, g, prev_pts, ransac_px)
            if h is not None and ni >= min_inliers:
                steps[i - 1], inl[i - 1], res[i - 1] = h, ni, r
        prev_g = g
        if i < m:                                  # last frame: target only
            prev_pts = stars.detect(frame, max_stars=max_stars)[:, :2]
    return a, steps, inl, res


def estimate(path, *, max_stars: int = 800, ransac_px: float = 0.7,
             min_inliers: int = 12, smooth: int = 3, max_dev: float = 0.25,
             progress=None, workers: int | None = None,
             chunk: int | None = None) -> Track:
    """Track sky motion between every adjacent pair of frames.

    Stars are detected in frame k and followed into k+1 with pyramidal LK;
    a RANSAC homography rejects static foreground points (they don't move
    with the sky) and mis-tracks. Steps with too few inliers are filled from
    their neighbours, as are steps whose motion is more than `max_dev` off
    the clip median (a speed-ramped clip would need this loosened). Steps
    are then box-smoothed over ±`smooth` frames —
    timelapse intervals are regular, so the true step varies slowly while the
    per-step fit noise does not.

    Steps are independent, so the clip is cut into contiguous chunks tracked
    by several processes (`hw.plan`, or `workers`); each chunk is decoded
    sequentially from a single seek. `progress(done, total)` reports steps
    completed.
    """
    from . import hw
    info = video.probe(path)
    n = info.frame_count
    steps = np.full((n - 1, 3, 3), np.nan)
    inl = np.zeros(n - 1, int)
    res = np.full(n - 1, np.nan)

    workers = hw.plan(info.width, info.height,
                      workers=workers).track_workers
    if chunk is None:        # a few chunks per worker keeps progress smooth
        chunk = max(8, math.ceil((n - 1) / (workers * 4)))
    spans = [(a, min(a + chunk, n - 1)) for a in range(0, n - 1, chunk)]
    args = (max_stars, ransac_px, min_inliers)
    done = 0

    def take(r):
        nonlocal done
        a, s, i_, r_ = r
        steps[a:a + len(s)], inl[a:a + len(s)], res[a:a + len(s)] = s, i_, r_
        done += len(s)
        if progress:
            progress(done, n - 1)

    if workers == 1 or len(spans) == 1:
        for a, b in spans:
            take(_steps_chunk(path, a, b, *args))
    else:
        with ProcessPoolExecutor(min(workers, len(spans))) as ex:
            futs = [ex.submit(_steps_chunk, str(path), a, b, *args)
                    for a, b in spans]
            for f in as_completed(futs):
                take(f.result())

    return finalize(steps, inl, res, info.width, info.height,
                    smooth=smooth, max_dev=max_dev)


def finalize(steps: np.ndarray, inl: np.ndarray, res: np.ndarray,
             w: int, h: int, *, smooth: int = 3,
             max_dev: float = 0.25) -> Track:
    """Reject, gap-fill and smooth raw per-step fits (NaN steps = no fit)."""
    n = len(steps) + 1
    steps, inl = steps.copy(), inl.copy()
    good = ~np.isnan(steps[:, 0, 0])
    if good.any():
        # a washed-out stretch (dawn, moonrise) loses the stars and the fit
        # locks onto the static foreground; regular intervals mean the true
        # step motion barely varies, so reject steps far from the clip median
        tmp = Track(steps=np.where(good[:, None, None], steps, np.eye(3)),
                    inliers=inl, residual_px=math.nan)
        mot = np.array([tmp.motion_px(k, w, h) for k in range(n - 1)])
        med = float(np.median(mot[good]))
        good &= np.abs(mot - med) <= max_dev * med
        inl[~good] = 0
    if good.sum() < max(3, (n - 1) // 4):
        raise RuntimeError("sky tracking failed — too few frames with stars")
    idx = np.arange(n - 1)
    flat = steps.reshape(n - 1, 9)
    for j in range(9):                           # fill gaps by interpolation
        flat[~good, j] = np.interp(idx[~good], idx[good], flat[good, j])

    if smooth > 0:
        kern = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        padded = np.pad(flat, ((smooth, smooth), (0, 0)), mode="edge")
        flat = np.column_stack([np.convolve(padded[:, j], kern, "valid")
                                for j in range(9)])

    return Track(steps=flat.reshape(n - 1, 3, 3), inliers=inl,
                 residual_px=float(np.median(res[good])))
