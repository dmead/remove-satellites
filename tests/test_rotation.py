import math

from remote_satellites.core import rotation

from .synth import make_clip


def test_recovers_center_and_rate(tmp_path):
    clip = tmp_path / "spin.mp4"
    truth = make_clip(clip, n=60, center=(500.0, 360.0), total_deg=12.0)

    m = rotation.estimate(clip, ref_index=0, samples=18)

    assert math.hypot(m.center[0] - 500.0, m.center[1] - 360.0) < 8.0
    assert abs(m.total_deg - 12.0) < 0.6
    assert m.residual_px < 2.0
    assert m.n_pairs >= 8


def test_no_rotation_is_stable(tmp_path):
    clip = tmp_path / "still.mp4"
    make_clip(clip, n=40, total_deg=0.0, trail=None)
    m = rotation.estimate(clip, ref_index=0, samples=12)
    # a static field should report ~zero spin without blowing up
    assert abs(m.total_deg) < 1.0
