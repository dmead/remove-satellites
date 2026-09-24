"""GUI smoke test: drive the real MainWindow offscreen on a synthetic clip.

Only modal dialogs are stubbed; the slots, worker threads and cleaner are the
shipping code. Locks the port to the tracking engine: auto-detect must yield
a track + foreground mask, the After preview must remove the trail and keep
the land, and export must write every frame.
"""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from remove_satellites.core import video                    # noqa: E402

from .synth import make_clip                                 # noqa: E402
from .test_pipeline import _line_median                      # noqa: E402


def _wait(app, cond, timeout=300):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "timed out"


def test_gui_detect_preview_export(tmp_path, monkeypatch):
    from remove_satellites.gui import window as W

    clip = tmp_path / "land.mp4"
    trail, g = (25, 31), 100
    truth = make_clip(clip, n=60, total_deg=0.0, drift=(3.0, -1.0),
                      trail=trail, ground=g)
    h = truth["h"]
    msgs = []
    monkeypatch.setattr(W.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(clip), "")))
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, _n=name: msgs.append((_n, a[2]))))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = W.MainWindow()
    win._open()
    win.detect_btn.click()
    _wait(app, lambda: win.detect_btn.isEnabled())
    assert win.track is not None and win.fg is not None, msgs
    assert "px/frame" in win.rot_status.text()
    assert win.fg[h - g + 8:].mean() > 0.95              # land found
    assert (tmp_path / "land.track.npz").exists()        # shared cache

    k = 27
    win.scrub.setValue(k)
    win._set_view(True)
    _wait(app, lambda: k in win._clean_cache)
    out = win._clean_cache[k]
    raw = video.read_frame(clip, k)
    assert _line_median(raw, k, trail, truth["w"], h) > 180
    assert _line_median(out, k, trail, truth["w"], h) < 80
    land = slice(h - g + 8, h)
    assert np.abs(out[land].astype(int) - raw[land].astype(int)).mean() < 3

    dst = tmp_path / "out.mp4"
    win.out_edit.setText(str(dst))
    win.export_btn.click()
    _wait(app, lambda: win._job is None and not win.progress.isVisible())
    assert ("information", f"Saved:\n{dst}") in msgs
    assert video.probe(dst).frame_count == 60
    win.close()

