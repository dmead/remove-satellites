"""Background QThread workers so the UI never blocks on OpenCV/ffmpeg work."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core import pipeline, rotation
from ..core.rotation import RotationModel


class DetectWorker(QThread):
    progress = Signal(int, int)
    done = Signal(object)          # RotationModel
    failed = Signal(str)

    def __init__(self, path, ref=0, samples=25):
        super().__init__()
        self.path, self.ref, self.samples = path, ref, samples

    def run(self):
        try:
            m = rotation.estimate(self.path, ref_index=self.ref,
                                  samples=self.samples,
                                  progress=lambda i, n: self.progress.emit(i, n))
            self.done.emit(m)
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(str(e))


class FrameWorker(QThread):
    """Compute one cleaned frame for the before/after preview."""
    done = Signal(int, object)     # index, ndarray
    failed = Signal(str)

    def __init__(self, path, model: RotationModel, radius: int, index: int):
        super().__init__()
        self.path, self.model = path, model
        self.radius, self.index = radius, index

    def run(self):
        try:
            cleaner = pipeline.Cleaner(self.path, self.model, radius=self.radius)
            frame = cleaner.frame(self.index)
            self.done.emit(self.index, frame)
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(str(e))


class ExportWorker(QThread):
    progress = Signal(int, int)
    done = Signal(str)             # output path
    failed = Signal(str)

    def __init__(self, path, model: RotationModel, radius: int, out_path,
                 crf: int = 16):
        super().__init__()
        self.path, self.model = path, model
        self.radius, self.out_path, self.crf = radius, out_path, crf
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            cleaner = pipeline.Cleaner(self.path, self.model, radius=self.radius)
            cleaner.run(self.out_path, crf=self.crf,
                        progress=lambda k, n: self.progress.emit(k, n),
                        cancel=lambda: self._cancel)
            if self._cancel:
                self.failed.emit("cancelled")
            else:
                self.done.emit(str(self.out_path))
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(str(e))
