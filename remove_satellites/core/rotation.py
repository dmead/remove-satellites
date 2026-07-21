"""Estimate the sky's rigid rotation (pole location + angular rate).

A fixed-tripod star field rotates uniformly about the celestial pole. We fit
that model from star centroids: for several baselines from a reference frame
we recover a rigid (rotation+translation) transform with RANSAC, turn each
into an angle and a fixed point (the pole), then combine them —
    angle(k) = omega * (k - ref_index)      [radians, linear in frame index]
    center   = robust mean of the fixed points
Robustness comes from RANSAC per pair plus weighting well-conditioned
(larger-rotation) pairs more heavily.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import stars, video


@dataclass(frozen=True)
class RotationModel:
    center: tuple[float, float]   # (cx, cy) pole, pixels
    omega: float                  # radians per frame (signed)
    ref_index: int
    n_frames: int
    residual_px: float            # median alignment residual across fits
    n_pairs: int                  # usable baselines that fit cleanly

    def angle_deg(self, frame_index: int) -> float:
        """Sky rotation (degrees) at `frame_index` relative to the reference."""
        return np.degrees(self.omega * (frame_index - self.ref_index))

    @property
    def total_deg(self) -> float:
        return abs(np.degrees(self.omega * (self.n_frames - 1)))


def _fixed_point(m: np.ndarray) -> tuple[float, float] | None:
    """Fixed point of a 2x3 similarity transform (its centre of rotation)."""
    a, b = m[0, 0], m[1, 0]              # a = s*cos, b = s*sin
    tx, ty = m[0, 2], m[1, 2]
    # (a-1)cx - b cy = -tx ; b cx + (a-1) cy = -ty
    A = np.array([[a - 1.0, -b], [b, a - 1.0]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    cx, cy = np.linalg.solve(A, np.array([-tx, -ty]))
    return float(cx), float(cy)


def _match(ref_pts: np.ndarray, tgt_pts: np.ndarray, max_dist: float):
    """Nearest-neighbour putative matches ref->tgt within max_dist."""
    if len(ref_pts) < 3 or len(tgt_pts) < 3:
        return None, None
    tree = cKDTree(tgt_pts)
    dist, idx = tree.query(ref_pts, distance_upper_bound=max_dist)
    ok = np.isfinite(dist)
    if ok.sum() < 6:
        return None, None
    return ref_pts[ok], tgt_pts[idx[ok]]


def estimate(path, *, ref_index: int = 0, samples: int = 25,
             max_stars: int = 120, match_dist: float = 60.0,
             progress=None) -> RotationModel:
    info = video.probe(path)
    n = info.frame_count
    ref_index = int(np.clip(ref_index, 0, n - 1))

    ref = stars.detect(video.read_frame(path, ref_index), max_stars=max_stars)
    if len(ref) < 6:
        raise RuntimeError("too few stars detected in the reference frame")
    ref_xy = ref[:, :2]

    # baselines spread across the clip (skip tiny ones — angle too small to fit)
    offsets = np.unique(np.linspace(0, n - 1, samples + 1).astype(int))
    offsets = offsets[offsets != ref_index]

    angles: list[float] = []
    dks: list[float] = []
    weights: list[float] = []
    centers: list[tuple[float, float]] = []
    cweights: list[float] = []
    resids: list[float] = []

    for j, k in enumerate(offsets):
        if progress:
            progress(j, len(offsets))
        tgt = stars.detect(video.read_frame(path, int(k)), max_stars=max_stars)
        if len(tgt) < 6:
            continue
        src, dst = _match(ref_xy, tgt[:, :2], match_dist)
        if src is None:
            continue
        m, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=5000, confidence=0.999)
        if m is None or inliers is None:
            continue
        n_in = int(inliers.sum())
        if n_in < 8:
            continue
        theta = float(np.arctan2(m[1, 0], m[0, 0]))
        dk = float(k - ref_index)
        angles.append(theta)
        dks.append(dk)
        weights.append(float(n_in))
        # residual of inlier matches after transform (alignment quality)
        proj = (m[:, :2] @ src.T).T + m[:, 2]
        r = np.hypot(*(proj - dst).T)[inliers.ravel().astype(bool)]
        resids.append(float(np.median(r)))
        c = _fixed_point(m)
        if c is not None and abs(theta) > np.radians(0.5):
            centers.append(c)
            cweights.append(n_in * abs(theta))  # big, star-rich baselines win

    if len(angles) < 3:
        raise RuntimeError("rotation fit failed — not enough consistent baselines")

    dks_a = np.array(dks)
    ang_a = np.array(angles)
    w_a = np.array(weights)
    # least-squares slope through the origin: omega = Σ w·θ·dk / Σ w·dk²
    omega = float(np.sum(w_a * ang_a * dks_a) / np.sum(w_a * dks_a**2))

    if centers:
        cen = np.array(centers)
        cw = np.array(cweights)
        center = (float(np.average(cen[:, 0], weights=cw)),
                  float(np.average(cen[:, 1], weights=cw)))
    else:  # degenerate: rotation too small to locate a pole
        center = (info.width / 2.0, info.height / 2.0)

    return RotationModel(
        center=center, omega=omega, ref_index=ref_index, n_frames=n,
        residual_px=float(np.median(resids)) if resids else float("nan"),
        n_pairs=len(angles))
