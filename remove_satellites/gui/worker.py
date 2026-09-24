"""Background QThread workers so the UI never blocks on OpenCV/ffmpeg work."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..core import foreground, pipeline, track, video


class DetectWorker(QThread):
    """Track the sky motion (cached next to the clip) and find the
    camera-fixed foreground. Emits a dict: track, fg (mask), summary,
    cached (bool)."""
    progress = Signal(int, int)
    stage = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            cache = track.default_cache(self.path)
            self.stage.emit("Loading cached sky track…" if cache.exists()
                            else "Tracking sky motion…")
            tr, cached = track.load_or_estimate(
                self.path, cache,
                progress=lambda i, n: self.progress.emit(i, n))
            self.stage.emit("Finding foreground…")
            fg = foreground.static_mask(self.path, tr)
            info = video.probe(self.path)
            self.stage.emit("Estimating a pole for the manual override…")
            self.done.emit(dict(track=tr, fg=fg, cached=cached,
                                pole_guess=pole_guess(self.path, tr, info),
                                summary=track.summary(tr, info.width,
                                                      info.height)))
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(str(e))


def pole_guess(path, tr, info) -> tuple[float, float, float] | None:
    """(pole x, pole y, total spin deg) to prefill the manual override.

    The classic pole + rate fit is exactly what the override means, so use
    it when it agrees with the tracked motion; when it clearly doesn't (a
    far-off-frame pole: it collapses to ~no rotation), fall back to the
    best-fit similarity of the tracked whole-clip motion.
    """
    import math

    import cv2
    import numpy as np

    from ..core import rotation
    w, h, n = info.width, info.height, info.frame_count
    tracked = float(np.median([tr.motion_px(k, w, h)
                               for k in range(0, n - 1, max(1, n // 50))]))
    try:
        m = rotation.estimate(path)
        cx, cy = m.center
        r = max(math.hypot(x - cx, y - cy) for x, y in
                [(0, 0), (w, 0), (0, h), (w, h)])
        # corner motion per frame implied by the pole model
        if abs(m.omega) * r >= 0.5 * tracked:
            return float(cx), float(cy), float(m.total_deg * np.sign(m.omega))
    except Exception:                                           # noqa: BLE001
        pass
    good = np.flatnonzero(tr.inliers > 0)
    a, b = (int(good[0]), int(good[-1]) + 1) if good.size else (0, n - 1)
    xs, ys = np.meshgrid(np.linspace(0, w, 12), np.linspace(0, h, 8))
    p = np.column_stack([xs.ravel(), ys.ravel()])
    q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), tr.between(a, b))
    sim, _ = cv2.estimateAffinePartial2D(p, q.reshape(-1, 2))
    fp = rotation._fixed_point(sim) if sim is not None else None
    if fp is None:
        return None
    ang = math.degrees(math.atan2(sim[1, 0], sim[0, 0])) * (n - 1) / max(1, b - a)
    return fp[0], fp[1], ang


class FrameWorker(QThread):
    """Compute one cleaned frame for the before/after preview."""
    done = Signal(int, object)     # index, ndarray
    failed = Signal(str)

    def __init__(self, path, motion, fg_mask, radius: int, index: int):
        super().__init__()
        self.path, self.motion, self.fg = path, motion, fg_mask
        self.radius, self.index = radius, index

    def run(self):
        try:
            cleaner = pipeline.Cleaner(self.path, self.motion,
                                       radius=self.radius, fg_mask=self.fg)
            self.done.emit(self.index, cleaner.frame(self.index))
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(str(e))


class ExportWorker(QThread):
    progress = Signal(int, int)
    done = Signal(str)             # output path
    failed = Signal(str)

    def __init__(self, path, motion, fg_mask, radius: int, out_path,
                 crf: int = 16):
        super().__init__()
        self.path, self.motion, self.fg = path, motion, fg_mask
        self.radius, self.out_path, self.crf = radius, out_path, crf
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            cleaner = pipeline.Cleaner(self.path, self.motion,
                                       radius=self.radius, fg_mask=self.fg)
            cleaner.run(self.out_path, crf=self.crf,
                        progress=lambda k, n: self.progress.emit(k, n),
                        cancel=lambda: self._cancel)
            if self._cancel:
                self.failed.emit("cancelled")
            else:
                self.done.emit(str(self.out_path))
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(str(e))
