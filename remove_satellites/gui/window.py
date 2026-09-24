"""remove-satellites main window — open a clip, track the sky motion, preview
before/after, and export the trail-free video.

Auto-detect runs the same engine as the CLI (homography tracking, cached as
`<clip>.track.npz`, plus the foreground mask). The old pole + rate controls
survive as a manual override for clips the tracker can't handle."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSlider,
                               QSpinBox, QStyle, QToolButton, QVBoxLayout,
                               QWidget)

from ..core import track, video
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
        self.track = None                    # auto-detected Track
        self.fg = None                       # foreground mask (1 = land)
        self.manual: RotationModel | None = None   # manual pole override
        self._rev = 0                        # bumps on any motion change
        self.preset_output = preset_output

        self._worker = None                 # keep refs so QThreads aren't GC'd
        self._job = None                     # currently running cancellable render
        self._clean_cache: dict[int, object] = {}
        self._playing = False
        # a fully-rendered cleaned copy for smooth After playback
        self._clean_video: str | None = None
        self._clean_key = None
        self._pending_clean = None           # (tmp, key, then_play)
        self._tmpdir = tempfile.mkdtemp(prefix="remove-sat-")

        self._build_ui()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

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

        def _icon(sp):
            return self.style().standardIcon(sp)

        self.rewind_btn = QToolButton()
        self.rewind_btn.setIcon(_icon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.rewind_btn.setToolTip("Rewind to start")
        self.rewind_btn.clicked.connect(self._rewind)
        self.play_btn = QToolButton()
        self._play_icon = _icon(QStyle.StandardPixmap.SP_MediaPlay)
        self._pause_icon = _icon(QStyle.StandardPixmap.SP_MediaPause)
        self.play_btn.setIcon(self._play_icon)
        self.play_btn.setToolTip("Play / pause")
        self.play_btn.clicked.connect(self._toggle_play)
        viewrow.addWidget(self.rewind_btn)
        viewrow.addWidget(self.play_btn)
        viewrow.addSpacing(12)

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
        g = QGroupBox("Sky motion")
        lay = QVBoxLayout(g)
        self.detect_btn = QPushButton("Auto-detect sky motion")
        self.detect_btn.clicked.connect(self._detect)
        lay.addWidget(self.detect_btn)

        self.rot_status = QLabel("")
        self.rot_status.setWordWrap(True)
        self.rot_status.setStyleSheet("color:#888;")
        lay.addWidget(self.rot_status)

        self.show_fg = QCheckBox("Show foreground mask")
        self.show_fg.toggled.connect(self._refresh_preview)
        lay.addWidget(self.show_fg)

        # the old pole + rate model, kept as a fallback
        self.override = QGroupBox("Manual pole override")
        self.override.setCheckable(True)
        self.override.setChecked(False)
        self.override.toggled.connect(self._on_override)
        form = QFormLayout(self.override)
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
        self.show_pole = QCheckBox("Show pole marker")
        self.show_pole.setChecked(True)
        self.show_pole.toggled.connect(self._refresh_preview)
        form.addRow(self.show_pole)
        lay.addWidget(self.override)
        return g

    @property
    def motion(self):
        """What the cleaner uses: the manual override when it's on,
        otherwise the tracked sky motion."""
        if self.override.isChecked() and self.manual is not None:
            return self.manual
        return self.track

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
        for w in (self.detect_btn, self.override, self.show_fg,
                  self.radius, self.export_btn, self.scrub, self.frame_spin,
                  self.before_btn, self.after_btn, self.play_btn,
                  self.rewind_btn):
            w.setEnabled(on)

    def _open(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open timelapse", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if not fn:
            return
        self._stop_play()
        try:
            self.info = video.probe(fn)
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.critical(self, "remove-satellites", f"Cannot open:\n{e}")
            return
        self.path = fn
        self.track = self.fg = self.manual = None
        self.override.setChecked(False)
        self._invalidate_motion()
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
        self._display()
        if track.default_cache(fn).exists():      # cheap: load it right away
            self._detect()
        else:
            self.rot_status.setText(
                "Run auto-detect: tracks the stars through every frame "
                "(a few minutes for 4K; cached for next time).")

    # ---- sky motion -----------------------------------------------------
    def _detect(self):
        if not self.path:
            return
        self.detect_btn.setEnabled(False)
        self.rot_status.setText("Detecting…")
        self._worker = DetectWorker(self.path)
        self._worker.stage.connect(self.rot_status.setText)
        self._worker.progress.connect(
            lambda i, n: self.rot_status.setText(
                f"Tracking sky motion… frame {i} of {n}"))
        self._worker.done.connect(self._detected)
        self._worker.failed.connect(self._detect_failed)
        self._worker.start()

    @Slot(object)
    def _detected(self, res: dict):
        self.detect_btn.setEnabled(True)
        self.track, self.fg = res["track"], res["fg"]
        self._prefill_override(res.get("pole_guess"))
        self._invalidate_motion()
        self.rot_status.setText(
            f"{res['summary']} · foreground {100 * self.fg.mean():.0f}% of "
            f"frame{' · cached track' if res['cached'] else ''}")
        self._refresh_preview()

    @Slot(str)
    def _detect_failed(self, msg: str):
        self.detect_btn.setEnabled(True)
        self.rot_status.setText("Detection failed — try the manual pole override.")
        QMessageBox.warning(self, "remove-satellites",
                            f"Sky-motion detection failed:\n{msg}")

    def _prefill_override(self, guess):
        """Start the manual override from a pole estimate (see
        worker.pole_guess), not (0, 0, 0°) — which silently means 'no
        rotation'. Far-off-frame poles give big numbers; that's honest."""
        self.manual = None                 # rebuilt from the spins if enabled
        if guess is None:
            return
        block = (self.cx_spin, self.cy_spin, self.total_spin)
        for w in block:
            w.blockSignals(True)
        self.cx_spin.setValue(guess[0])
        self.cy_spin.setValue(guess[1])
        self.total_spin.setValue(guess[2])
        for w in block:
            w.blockSignals(False)

    def _on_manual_rotation(self):
        if not self.info:
            return
        n = self.info.frame_count
        total_rad = math.radians(self.total_spin.value())
        omega = total_rad / (n - 1) if n > 1 else 0.0
        self.manual = RotationModel(
            center=(self.cx_spin.value(), self.cy_spin.value()),
            omega=omega, ref_index=0, n_frames=n,
            residual_px=float("nan"), n_pairs=0)
        if self.override.isChecked():
            self._invalidate_motion()
            self._refresh_preview()

    def _on_override(self, on: bool):
        if on and self.manual is None:
            self._on_manual_rotation()
        self._invalidate_motion()
        if on:
            self.rot_status.setText("manual pole + rate (override)")
        self._refresh_preview()

    def _invalidate_motion(self):
        self._rev += 1
        self._invalidate_clean()

    # ---- preview --------------------------------------------------------
    def _on_scrub(self, v: int):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(v)
        self.frame_spin.blockSignals(False)
        # fast paths (raw, or a pre-rendered cleaned clip) draw immediately;
        # the slow on-demand single-frame clean is debounced while scrubbing
        if self._playing or not self.after_btn.isChecked() \
                or self._clean_video_valid():
            self._display(v)
        else:
            self._debounce.start()

    def _blit(self, frame):
        if self.show_fg.isChecked() and self.fg is not None:
            frame = preview.tint_mask(frame, self.fg)
        pm = preview.bgr_to_pixmap(frame)
        if self.override.isChecked() and self.show_pole.isChecked() \
                and self.manual is not None:
            pm = preview.draw_pole(pm, self.manual.center)
        self.preview.set_source(pm)

    def _display(self, idx: int | None = None):
        """Show the right frame for the current view (Before / After)."""
        if not self.path:
            return
        if idx is None:
            idx = self.scrub.value()
        after = self.after_btn.isChecked()
        if after and self.motion is not None and self._clean_video_valid():
            self._blit(video.read_frame(self._clean_video, idx))
        elif after and self.motion is not None:
            cached = self._clean_cache.get(idx)
            if cached is not None:
                self._blit(cached)
            else:                       # slow one-off clean of this frame
                self.rot_status_busy("cleaning preview frame…")
                self._worker = FrameWorker(self.path, self.motion, self.fg,
                                           self.radius.value(), idx)
                self._worker.done.connect(self._frame_ready)
                self._worker.failed.connect(lambda m: self.rot_status_busy(""))
                self._worker.start()
        else:
            self._blit(video.read_frame(self.path, idx))

    def _refresh_preview(self):
        self._display()

    def rot_status_busy(self, msg):
        self.statusBar().showMessage(msg) if msg else self.statusBar().clearMessage()

    @Slot(int, object)
    def _frame_ready(self, idx: int, frame):
        self.rot_status_busy("")
        self._clean_cache[idx] = frame
        if self.after_btn.isChecked() and self.scrub.value() == idx:
            self._blit(frame)

    # ---- playback -------------------------------------------------------
    def _toggle_play(self):
        self._stop_play() if self._playing else self._start_play()

    def _start_play(self):
        if not self.path:
            return
        # After playback needs a fully-rendered cleaned clip; render it once
        if self.after_btn.isChecked() and self.motion is not None \
                and not self._clean_video_valid():
            self._render_clean(then_play=True)
            return
        self._begin_play()

    def _begin_play(self):
        self._playing = True
        self.play_btn.setIcon(self._pause_icon)
        if self.scrub.value() >= self.scrub.maximum():
            self.scrub.setValue(0)
        fps = self.info.fps if self.info and self.info.fps else 24.0
        self._play_timer.start(int(1000 / max(1.0, fps)))

    def _stop_play(self):
        self._playing = False
        self._play_timer.stop()
        self.play_btn.setIcon(self._play_icon)

    def _play_tick(self):
        v = self.scrub.value() + 1
        if v > self.scrub.maximum():
            v = 0                       # loop
        self.scrub.setValue(v)

    def _rewind(self):
        self._stop_play()
        self.scrub.setValue(0)

    def _on_radius(self, v: int):
        self.radius_lbl.setText(str(v))
        self._invalidate_clean()
        if self.after_btn.isChecked():
            self._debounce.start()

    def _set_view(self, after: bool):
        self.before_btn.setChecked(not after)
        self.after_btn.setChecked(after)
        self._display()

    # ---- cleaned-clip cache for After playback --------------------------
    def _settings_key(self):
        return (self._rev, self.radius.value()) \
            if self.motion is not None else None

    def _clean_video_valid(self) -> bool:
        return bool(self._clean_video and self.motion is not None
                    and self._clean_key == self._settings_key()
                    and os.path.exists(self._clean_video))

    def _invalidate_clean(self):
        self._clean_cache.clear()
        if self._playing and self.after_btn.isChecked():
            self._stop_play()
        if self._clean_video and os.path.exists(self._clean_video):
            try:
                os.remove(self._clean_video)
            except OSError:
                pass
        self._clean_video = None
        self._clean_key = None

    def _render_clean(self, *, then_play: bool):
        tmp = os.path.join(self._tmpdir, f"clean_{abs(hash(self._settings_key()))}.mp4")
        self._pending_clean = (tmp, self._settings_key(), then_play)
        self._begin_progress("Rendering cleaned preview…")
        self._job = ExportWorker(self.path, self.motion, self.fg,
                                 self.radius.value(), tmp, crf=18)
        self._job.progress.connect(self._export_progress)
        self._job.done.connect(self._clean_ready)
        self._job.failed.connect(self._clean_failed)
        self._job.start()

    @Slot(str)
    def _clean_ready(self, out: str):
        self._end_progress()
        tmp, key, then_play = self._pending_clean
        self._pending_clean = None
        self._clean_video, self._clean_key = tmp, key
        self._display()
        if then_play:
            self._begin_play()

    @Slot(str)
    def _clean_failed(self, msg: str):
        self._end_progress()
        self._pending_clean = None
        if msg != "cancelled":
            QMessageBox.warning(self, "remove-satellites",
                                f"Preview render failed:\n{msg}")

    # ---- export ---------------------------------------------------------
    def _pick_output(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save cleaned video", self.out_edit.text(), "MP4 (*.mp4)")
        if fn:
            self.out_edit.setText(fn)

    def _export(self):
        if not (self.path and self.motion is not None):
            QMessageBox.information(
                self, "remove-satellites",
                "Auto-detect the sky motion first (or use the manual pole "
                "override).")
            return
        self._stop_play()
        out = self.out_edit.text().strip()
        if not out:
            self._pick_output()
            out = self.out_edit.text().strip()
            if not out:
                return
        self._begin_progress("Exporting cleaned video…")
        self._job = ExportWorker(self.path, self.motion, self.fg,
                                 self.radius.value(), out,
                                 crf=self.crf.currentData())
        self._job.progress.connect(self._export_progress)
        self._job.done.connect(self._export_done)
        self._job.failed.connect(self._export_failed)
        self._job.start()

    # ---- shared progress UI (export + preview render) -------------------
    _BUSY_WIDGETS = ("export_btn", "open_btn", "play_btn", "detect_btn",
                     "override", "radius")

    def _begin_progress(self, label: str):
        self.rot_status_busy(label)
        self.progress.setValue(0)
        self.progress.setMaximum(self.info.frame_count)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(True)
        for name in self._BUSY_WIDGETS:
            getattr(self, name).setEnabled(False)

    def _end_progress(self):
        self.rot_status_busy("")
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        for name in self._BUSY_WIDGETS:
            getattr(self, name).setEnabled(True)
        self._job = None

    @Slot(int, int)
    def _export_progress(self, k: int, n: int):
        self.progress.setMaximum(n)
        self.progress.setValue(k)

    def _cancel_export(self):
        if self._job is not None:
            self._job.cancel()
            self.cancel_btn.setEnabled(False)

    @Slot(str)
    def _export_done(self, out: str):
        self._end_progress()
        QMessageBox.information(self, "remove-satellites", f"Saved:\n{out}")

    @Slot(str)
    def _export_failed(self, msg: str):
        self._end_progress()
        if msg != "cancelled":
            QMessageBox.warning(self, "remove-satellites", f"Export failed:\n{msg}")

    def closeEvent(self, ev):
        self._stop_play()
        if self._job is not None:
            self._job.cancel()
            self._job.wait(2000)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().closeEvent(ev)
