import cv2
import numpy as np

from remove_satellites.core import pipeline, rotation, stars, video

from .synth import make_clip


def _line_median(frame, k, trail, w, h):
    """Median brightness sampled along the exact synthetic streak at frame k."""
    t = (k - trail[0]) / max(1, trail[1] - trail[0] - 1)
    x = 60 + t * (w - 120)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ts = np.linspace(0, 1, 200)
    xs = (x + 40 * ts).astype(int)
    ys = (40 + (h - 80) * ts).astype(int)
    return float(np.median(gray[ys, xs]))


def test_trail_removed_stars_kept(tmp_path):
    clip = tmp_path / "sat.mp4"
    tr = (25, 31)
    truth = make_clip(clip, n=60, trail=tr)
    m = rotation.estimate(clip, ref_index=0, samples=18)

    cleaner = pipeline.Cleaner(clip, m, radius=6)
    k = 27  # mid-trail frame
    raw = video.read_frame(clip, k)
    clean = cleaner.frame(k)

    raw_line = _line_median(raw, k, tr, truth["w"], truth["h"])
    clean_line = _line_median(clean, k, tr, truth["w"], truth["h"])
    assert raw_line > 180          # the streak really is there in the source
    assert clean_line < 80         # and it's gone after cleaning

    # stars survive: essentially the same star count is detectable
    n_raw = len(stars.detect(raw))
    n_clean = len(stars.detect(clean))
    assert n_clean >= 0.8 * n_raw


def test_full_run_frame_count(tmp_path):
    clip = tmp_path / "sat.mp4"
    make_clip(clip, n=48)
    m = rotation.estimate(clip, ref_index=0, samples=14)
    out = tmp_path / "clean.mp4"
    n = pipeline.Cleaner(clip, m, radius=5).run(out)
    assert n == 48
    assert video.probe(out).frame_count == 48
