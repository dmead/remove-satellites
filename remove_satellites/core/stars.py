"""Star detection: bright, compact blobs → intensity-weighted centroids.

Deliberately simple and fast — we only need a few dozen reliable point
sources per frame to fit a rigid rotation, not a photometric catalogue.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def background(gray: np.ndarray, *, scale: int = 8, ksize: int = 7
               ) -> np.ndarray:
    """Smooth sky background: median of a downsampled copy, scaled back up.

    Median-filtering at 1/`scale` resolution spans ~scale*ksize full-res
    pixels — wide enough to ignore stars, narrow enough to follow horizon glow
    and light pollution gradients. Downsampling keeps it cheap at 4K.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // scale), max(1, h // scale)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small.astype(np.uint8), ksize).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def detect(frame: np.ndarray, *, max_stars: int = 120,
           k_sigma: float = 6.0, min_area: int = 2,
           max_area: int = 400) -> np.ndarray:
    """Return an (N, 3) array of [x, y, flux], brightest first.

    Subtract a local background, threshold the residual at
    median + k_sigma * MAD-sigma, label blobs, keep compact ones, take
    intensity-weighted centroids. The local background matters: a global
    threshold on a frame with horizon glow sits above every star.

    Speed matters (tracking runs this on every frame): the noise statistics
    come from a 1/16 subsample, blobs from OpenCV's connected components and
    the per-blob sums from one bincount each — ~10x faster at 4K than
    scipy.ndimage's per-label reductions, same 4-connected blobs.
    """
    gray = to_gray(frame).astype(np.float32)
    gray = gray - background(gray)
    sub = gray[::4, ::4]
    med = float(np.median(sub))
    mad = float(np.median(np.abs(sub - med))) or 1.0
    thresh = med + k_sigma * 1.4826 * mad

    mask = (gray > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=4, ltype=cv2.CV_32S)
    if n <= 1:
        return np.empty((0, 3), np.float32)

    lab = labels.ravel()
    on = np.flatnonzero(lab)
    li = lab[on]
    w = gray.ravel()[on] - med
    ys, xs = np.divmod(on, gray.shape[1])
    flux = np.bincount(li, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.bincount(li, weights=w * xs, minlength=n) / flux
        cy = np.bincount(li, weights=w * ys, minlength=n) / flux
    areas = stats[:, cv2.CC_STAT_AREA]

    keep = (areas >= min_area) & (areas <= max_area)
    keep[0] = False                                  # label 0 = background
    cx, cy, flux = cx[keep], cy[keep], flux[keep]
    if cx.size == 0:
        return np.empty((0, 3), np.float32)

    order = np.argsort(flux)[::-1][:max_stars]
    return np.column_stack([cx[order], cy[order], flux[order]]).astype(np.float32)
