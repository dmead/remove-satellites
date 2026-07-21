"""Convert OpenCV BGR frames to Qt pixmaps, with an optional pole marker."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    # copy() detaches from the numpy buffer so it can be freed safely
    return QPixmap.fromImage(img.copy())


def draw_pole(pixmap: QPixmap, center: tuple[float, float]) -> QPixmap:
    """Return a copy of `pixmap` with a crosshair drawn at `center` (image px)."""
    out = QPixmap(pixmap)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(255, 80, 80), 2)
    p.setPen(pen)
    cx, cy = center
    r = 14
    p.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
    p.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))
    p.drawEllipse(int(cx - r), int(cy - r), 2 * r, 2 * r)
    p.end()
    return out
