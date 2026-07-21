"""remove-satellites main window — open a clip, detect/adjust the sky rotation,
preview before/after, and export the trail-free video."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from ..core import rotation, video
from ..core.rotation import RotationModel
from . import preview
from .worker import DetectWorker, ExportWorker, FrameWorker


class PreviewLabel(QLabel):
    """A QLabel that keeps a source pixmap and rescales it to fit on resize."""

    def __init__(self):
        super().__init__()
        self._src = None
        self.setMinimumSize(480, 270)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#101014;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setText("Open a star-field timelapse to begin")

    def set_source(self, pixmap):
        self._src = pixmap
        self._rescale()

    def _rescale(self):
        if self._src is None:
            return
        super().setPixmap(self._src.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, ev):
        self._rescale()
        super().resizeEvent(ev)


class MainWindow(QMainWindow):
    def __init__(self, preset_output: str | None = None):
        super().__init__()
        self.setWindowTitle("remove-satellites — satellite-trail remover")
        self.resize(1180, 720)

        self.path: str | None = None
        self.info: video.VideoInfo | None = None
        self.model: RotationModel | None = None
        self.preset_output = preset_output

        self._worker = None                 # keep refs so QThreads aren't GC'd
        self._raw_frame = None
        self._clean_cache: dict[int, object] = {}

        self._build_ui()
        self._set_enabled(False)

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left: preview + scrubber
        left = QVBoxLayout()
        self.preview = PreviewLabel()
        left.addWidget(self.preview, 1)

        viewrow = QHBoxLayout()
        self.before_btn = QPushButton("Before")
        self.after_btn = QPushButton("After")
        for b in (self.before_btn, self.after_btn):
            b.setCheckable(True)
        self.before_btn.setChecked(True)
        self.before_btn.clicked.connect(lambda: self._set_view(False))
        self.after_btn.clicked.connect(lambda: self._set_view(True))
        viewrow.addWidget(self.before_btn)
        viewrow.addWidget(self.after_btn)
        viewrow.addSpacing(16)
        viewrow.addWidget(QLabel("Frame"))
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.valueChanged.connect(self._on_scrub)
        viewrow.addWidget(self.scrub, 1)
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.scrub.setValue)
        viewrow.addWidget(self.frame_spin)
        left.addLayout(viewrow)
        root.addLayout(left, 1)

        # right: controls
        panel = QVBoxLayout()
        panel.setSpacing(10)

        self.open_btn = QPushButton("Open video…")
        self.open_btn.clicked.connect(self._open)
        panel.addWidget(self.open_btn)

        self.info_lbl = QLabel("No file loaded")
        self.info_lbl.setWordWrap(True)
        panel.addWidget(self.info_lbl)

        panel.addWidget(self._rotation_group())
        panel.addWidget(self._cleaning_group())
        panel.addWidget(self._export_group())
        panel.addStretch(1)

        panelw = QWidget()
        panelw.setLayout(panel)
        panelw.setFixedWidth(340)
        root.addWidget(panelw)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._refresh_preview)

    def _rotation_group(self) -> QGroupBox:
        g = QGroupBox("Sky rotation")
        lay = QVBoxLayout(g)
        self.detect_btn = QPushButton("Auto-detect rotation")
        self.detect_btn.clicked.connect(self._detect)
        lay.addWidget(self.detect_btn)

        form = QFormLayout()
        self.cx_spin = QDoubleSpinBox()
        self.cy_spin = QDoubleSpinBox()
        for s in (self.cx_spin, self.cy_spin):
            s.setRange(-10000, 10000)
            s.setDecimals(1)
            s.valueChanged.connect(self._on_manual_rotation)
        self.total_spin = QDoubleSpinBox()
        self.total_spin.setRange(-180, 180)
        self.total_spin.setDecimals(3)
        self.total_spin.setSuffix(" °")
        self.total_spin.valueChanged.connect(self._on_manual_rotation)
        form.addRow("Pole X", self.cx_spin)
        form.addRow("Pole Y", self.cy_spin)
        form.addRow("Total spin", self.total_spin)
        lay.addLayout(form)

        self.show_pole = QCheckBox("Show pole marker")
        self.show_pole.setChecked(True)
        self.show_pole.toggled.connect(self._refresh_preview)
        lay.addWidget(self.show_pole)

        self.rot_status = QLabel("")
        self.rot_status.setStyleSheet("color:#888;")
        lay.addWidget(self.rot_status)
        return g

    def _cleaning_group(self) -> QGroupBox:
        g = QGroupBox("Cleaning")
        lay = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("Median window ±"))
        self.radius = QSlider(Qt.Orientation.Horizontal)
        self.radius.setRange(1, 30)
        self.radius.setValue(10)
        self.radius.valueChanged.connect(self._on_radius)
        row.addWidget(self.radius, 1)
        self.radius_lbl = QLabel("10")
        row.addWidget(self.radius_lbl)
        lay.addLayout(row)
        hint = QLabel("larger window = stronger trail rejection")
        hint.setStyleSheet("color:#888;")
        lay.addWidget(hint)
        return g

    def _export_group(self) -> QGroupBox:
        g = QGroupBox("Export")
        lay = QVBoxLayout(g)
        orow = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("output .mp4")
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._pick_output)
        orow.addWidget(self.out_edit, 1)
        orow.addWidget(browse)
        lay.addLayout(orow)

        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Quality"))
        self.crf = QComboBox()
        for v, name in [(14, "High (14)"), (16, "Good (16)"),
                        (18, "Medium (18)"), (20, "Small (20)")]:
            self.crf.addItem(name, v)
        self.crf.setCurrentIndex(1)
        qrow.addWidget(self.crf, 1)
        lay.addLayout(qrow)

        self.export_btn = QPushButton("Export cleaned video")
        self.export_btn.clicked.connect(self._export)
        lay.addWidget(self.export_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_export)
        lay.addWidget(self.cancel_btn)
        return g

    # ---- state ----------------------------------------------------------
    def _set_enabled(self, on: bool):
        for w in (self.detect_btn, self.cx_spin, self.cy_spin, self.total_spin,
                  self.radius, self.export_btn, self.scrub, self.frame_spin,
                  self.before_btn, self.after_btn):
            w.setEnabled(on)

    def _open(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open timelapse", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if not fn:
            return
        try:
            self.info = video.probe(fn)
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.critical(self, "remove-satellites", f"Cannot open:\n{e}")
            return
        self.path = fn
        self.model = None
        self._clean_cache.clear()
        n = self.info.frame_count
        self.scrub.setRange(0, n - 1)
        self.frame_spin.setRange(0, n - 1)
        self.scrub.setValue(n // 2)
        self.info_lbl.setText(
            f"<b>{Path(fn).name}</b><br>{self.info.width}×{self.info.height}"
            f" · {n} frames · {self.info.fps:.3g} fps"
            f" · {self.info.duration:.1f}s")
        default_out = self.preset_output or str(
            Path(fn).with_name(Path(fn).stem + "_no-satellites.mp4"))
        self.out_edit.setText(default_out)
        self._set_enabled(True)
        self.rot_status.setText("Run auto-detect (or set the pole manually).")
        self._show_raw()

    # ---- rotation -------------------------------------------------------
    def _detect(self):
        if not self.path:
            return
        self.detect_btn.setEnabled(False)
        self.rot_status.setText("Detecting…")
        self._worker = DetectWorker(self.path)
        self._worker.done.connect(self._detected)
        self._worker.failed.connect(self._detect_failed)
        self._worker.start()

    @Slot(object)
    def _detected(self, model: RotationModel):
        self.detect_btn.setEnabled(True)
        self._apply_model(model)
        self.rot_status.setText(
            f"pole ({model.center[0]:.0f}, {model.center[1]:.0f}) · "
            f"{model.total_deg:.2f}° total · residual {model.residual_px:.2f}px")

    @Slot(str)
    def _detect_failed(self, msg: str):
        self.detect_btn.setEnabled(True)
        self.rot_status.setText("Detection failed — set the pole manually.")
        QMessageBox.warning(self, "remove-satellites", f"Rotation detection failed:\n{msg}")

    def _apply_model(self, model: RotationModel):
        self.model = model
        self._clean_cache.clear()
        block = (self.cx_spin, self.cy_spin, self.total_spin)
        for w in block:
            w.blockSignals(True)
        self.cx_spin.setValue(model.center[0])
        self.cy_spin.setValue(model.center[1])
        self.total_spin.setValue(math.degrees(model.omega * (model.n_frames - 1)))
        for w in block:
            w.blockSignals(False)
        self._refresh_preview()

    def _on_manual_rotation(self):
        if not self.info:
            return
        n = self.info.frame_count
        total_rad = math.radians(self.total_spin.value())
        omega = total_rad / (n - 1) if n > 1 else 0.0
        self.model = RotationModel(
            center=(self.cx_spin.value(), self.cy_spin.value()),
            omega=omega, ref_index=0, n_frames=n,
            residual_px=float("nan"), n_pairs=0)
        self._clean_cache.clear()
        self.rot_status.setText("manual rotation")
        self._refresh_preview()

    # ---- preview --------------------------------------------------------
    def _on_scrub(self, v: int):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(v)
        self.frame_spin.blockSignals(False)
        self._raw_frame = None
        self._debounce.start()

    def _on_radius(self, v: int):
        self.radius_lbl.setText(str(v))
        self._clean_cache.clear()
        if self.after_btn.isChecked():
            self._debounce.start()

    def _set_view(self, after: bool):
        self.before_btn.setChecked(not after)
        self.after_btn.setChecked(after)
        self._refresh_preview()

    def _show_raw(self):
        if not self.path:
            return
        idx = self.scrub.value()
        if self._raw_frame is None:
            self._raw_frame = video.read_frame(self.path, idx)
        pm = preview.bgr_to_pixmap(self._raw_frame)
        if self.show_pole.isChecked() and self.model:
            pm = preview.draw_pole(pm, self.model.center)
        self.preview.set_source(pm)

    def _refresh_preview(self):
        if not self.path:
            return
        if self.after_btn.isChecked() and self.model:
            idx = self.scrub.value()
            cached = self._clean_cache.get(idx)
            if cached is not None:
                pm = preview.bgr_to_pixmap(cached)
                if self.show_pole.isChecked():
                    pm = preview.draw_pole(pm, self.model.center)
                self.preview.set_source(pm)
            else:
                self.rot_status_busy("cleaning preview frame…")
                self._worker = FrameWorker(self.path, self.model,
                                           self.radius.value(), idx)
                self._worker.done.connect(self._frame_ready)
                self._worker.failed.connect(lambda m: self.rot_status_busy(""))
                self._worker.start()
        else:
            self._show_raw()

    def rot_status_busy(self, msg):
        if msg:
            self.statusBar().showMessage(msg)
        else:
            self.statusBar().clearMessage()

    @Slot(int, object)
    def _frame_ready(self, idx: int, frame):
        self.rot_status_busy("")
        self._clean_cache[idx] = frame
        if self.after_btn.isChecked() and self.scrub.value() == idx:
            pm = preview.bgr_to_pixmap(frame)
            if self.show_pole.isChecked() and self.model:
                pm = preview.draw_pole(pm, self.model.center)
            self.preview.set_source(pm)

    # ---- export ---------------------------------------------------------
    def _pick_output(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save cleaned video", self.out_edit.text(), "MP4 (*.mp4)")
        if fn:
            self.out_edit.setText(fn)

    def _export(self):
        if not (self.path and self.model):
            QMessageBox.information(
                self, "remove-satellites",
                "Detect or set the sky rotation first.")
            return
        out = self.out_edit.text().strip()
        if not out:
            self._pick_output()
            out = self.out_edit.text().strip()
            if not out:
                return
        self.export_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setMaximum(self.info.frame_count)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self._worker = ExportWorker(self.path, self.model, self.radius.value(),
                                    out, crf=self.crf.currentData())
        self._worker.progress.connect(self._export_progress)
        self._worker.done.connect(self._export_done)
        self._worker.failed.connect(self._export_failed)
        self._worker.start()

    @Slot(int, int)
    def _export_progress(self, k: int, n: int):
        self.progress.setMaximum(n)
        self.progress.setValue(k)

    def _cancel_export(self):
        if isinstance(self._worker, ExportWorker):
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _end_export(self):
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.open_btn.setEnabled(True)

    @Slot(str)
    def _export_done(self, out: str):
        self._end_export()
        QMessageBox.information(self, "remove-satellites", f"Saved:\n{out}")

    @Slot(str)
    def _export_failed(self, msg: str):
        self._end_export()
        if msg != "cancelled":
            QMessageBox.warning(self, "remove-satellites", f"Export failed:\n{msg}")
