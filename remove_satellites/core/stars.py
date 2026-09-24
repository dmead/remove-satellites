"""Star detection: bright, compact blobs → intensity-weighted centroids.

Deliberately simple and fast — we only need a few dozen reliable point
sources per frame to fit a rigid rotation, not a photometric catalogue.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


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
    """
    gray = to_gray(frame).astype(np.float32)
    gray = gray - background(gray)
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med))) or 1.0
    sigma = 1.4826 * mad
    thresh = med + k_sigma * sigma

    mask = gray > thresh
    if not mask.any():
        return np.empty((0, 3), np.float32)

    labels, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 3), np.float32)

    areas = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
    flux = ndimage.sum(gray - med, labels, range(1, n + 1))
    cy, cx = np.array(
        ndimage.center_of_mass(gray - med, labels, range(1, n + 1))).T

    keep = (areas >= min_area) & (areas <= max_area)
    cx, cy, flux = cx[keep], cy[keep], flux[keep]
    if cx.size == 0:
        return np.empty((0, 3), np.float32)

    order = np.argsort(flux)[::-1][:max_stars]
    return np.column_stack([cx[order], cy[order], flux[order]]).astype(np.float32)
