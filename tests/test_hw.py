from remove_satellites.core import hw

GB = 1 << 30


def fake(monkeypatch, cores, free_gb):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: cores)
    monkeypatch.setattr(hw, "avail_bytes", lambda: int(free_gb * GB))
    monkeypatch.delenv("REMOVE_SATELLITES_WORKERS", raising=False)


def test_auto_plan_uses_the_cores(monkeypatch):
    fake(monkeypatch, 32, 100)
    p = hw.plan(3840, 2160, 10)
    assert p.export_workers == 32 // hw.EXPORT_THREADS
    assert p.export_workers * p.export_threads <= 32
    assert p.track_workers == hw.MAX_TRACK
    assert p.preview_threads == 32 and not p.overridden
    assert "32 CPUs" in p.describe()


def test_small_machine_and_little_ram(monkeypatch):
    fake(monkeypatch, 4, 3)                    # 4 cores, 3 GB free, 4K clip
    p = hw.plan(3840, 2160, 10)
    assert p.export_workers == 1               # one 4K window already ~1.8 GB
    assert p.export_threads == 4
    assert 1 <= p.track_workers <= 4
    assert p.cache_bytes <= 3 * GB // 4


def test_1080p_fits_more_export_processes_than_4k(monkeypatch):
    fake(monkeypatch, 32, 12)
    assert hw.plan(1920, 1080, 10).export_workers > \
        hw.plan(3840, 2160, 10).export_workers


def test_override_env_and_argument(monkeypatch):
    fake(monkeypatch, 32, 100)
    monkeypatch.setenv("REMOVE_SATELLITES_WORKERS", "3")
    p = hw.plan(1920, 1080)
    assert (p.track_workers, p.export_workers, p.overridden) == (3, 3, True)
    assert hw.plan(1920, 1080, workers=5).export_workers == 5   # arg wins
    fake(monkeypatch, 32, 2)                   # override still capped by RAM
    assert hw.plan(3840, 2160, 10, workers=16).export_workers == 1
