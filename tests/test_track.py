import cv2
import numpy as np

from remove_satellites.core import foreground, pipeline, stars, track, video

from .synth import make_clip
from .test_pipeline import _line_median


def test_track_follows_rotation(tmp_path):
    clip = tmp_path / "spin.mp4"
    make_clip(clip, n=40, center=(500.0, 360.0), total_deg=12.0, trail=None)
    tr = track.estimate(clip)
    # a star at a known spot should land where the true rotation puts it,
    # over a median-window-sized chain (the synth rounds star positions to
    # whole pixels, so long chains random-walk off by a few px)
    p0 = np.array([[[200.0, 150.0]]])
    k = 12
    ang = np.radians(12.0) * k / 39
    c, s = np.cos(ang), np.sin(ang)
    dx, dy = 200.0 - 500.0, 150.0 - 360.0
    truth = (500 + c * dx - s * dy, 360 + s * dx + c * dy)
    got = cv2.perspectiveTransform(p0, tr.between(0, k))[0, 0]
    assert np.hypot(*(got - truth)) < 1.5
    # and composing back returns home
    back = cv2.perspectiveTransform(p0, tr.between(k, 0) @ tr.between(0, k))
    assert np.hypot(*(back[0, 0] - p0[0, 0])) < 1e-6


def test_far_pole_pan_is_cleaned(tmp_path):
    """Pole at infinity (pure pan): the case the pole + rate model can't fit."""
    clip = tmp_path / "pan.mp4"
    tr_ = (25, 31)
    truth = make_clip(clip, n=60, total_deg=0.0, drift=(3.0, -1.0), trail=tr_)
    tr = track.estimate(clip)
    cleaner = pipeline.Cleaner(clip, tr, radius=6)
    k = 27
    raw, clean = video.read_frame(clip, k), cleaner.frame(k)
    assert _line_median(raw, k, tr_, truth["w"], truth["h"]) > 180
    assert _line_median(clean, k, tr_, truth["w"], truth["h"]) < 80
    assert len(stars.detect(clean)) >= 0.8 * len(stars.detect(raw))


def test_foreground_found_and_kept_sharp(tmp_path):
    clip = tmp_path / "land.mp4"
    g = 120
    truth = make_clip(clip, n=60, total_deg=0.0, drift=(3.0, -1.0),
                      trail=None, ground=g)
    h = truth["h"]
    tr = track.estimate(clip)
    mask = foreground.static_mask(clip, tr)
    assert mask[h - g + 8:].mean() > 0.95          # land found
    assert mask[40:h - g - 40].mean() < 0.05       # open sky left alone

    k = 30
    raw = video.read_frame(clip, k).astype(int)
    land = slice(h - g + 8, h)

    def err(out):
        return float(np.abs(out[land].astype(int) - raw[land]).mean())

    masked = pipeline.Cleaner(clip, tr, radius=6, fg_mask=mask).frame(k)
    unmasked = pipeline.Cleaner(clip, tr, radius=6).frame(k)
    # the land survives with the mask; star alignment alone smears it
    assert err(masked) < 3.0
    assert err(unmasked) > 3 * err(masked)


def test_no_foreground_on_open_sky(tmp_path):
    clip = tmp_path / "sky.mp4"
    make_clip(clip, n=40, total_deg=0.0, drift=(3.0, -1.0), trail=None)
    mask = foreground.static_mask(clip, track.estimate(clip))
    assert mask.mean() < 0.02
