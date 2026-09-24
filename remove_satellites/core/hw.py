"""How much of the machine to use — decided in one place, from the hardware.

Three kinds of parallel work, sized from logical CPUs, free RAM and the
clip's frame size:

* tracking — one process per chunk of frames; each holds two frames, so it
  scales to the core count (measured: 4K mono_east, 16 processes 35 s vs
  100 s sequential);
* export — contiguous chunks rendered by separate processes, each with a few
  threads. One 4K frame is memory-bandwidth bound and stops speeding up past
  ~8 threads, so more processes with fewer threads win. Measured on an
  i9-14900K (8P+16E, 32 threads), 4K clean + x264, frames/s:
  1x32 0.82 · 2x16 1.11 · 4x8 1.50 · 8x4 1.6-1.9 · 12x2 1.89 ·
  16x2 2.26 · 24x1 2.03 — hence EXPORT_THREADS = 2. Each 4K process was
  seen at ~2.5 GB plus ~0.5 GB for its encoder, so free RAM caps the count;
* preview — one frame at a time in the GUI: all cores as threads, plus a
  decoded-frame cache sized to free RAM.

Override the guess with --workers N (CLI) or REMOVE_SATELLITES_WORKERS=N,
which sets the process count for both tracking and export.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

EXPORT_THREADS = 2          # threads per export process (benchmark sweet spot)
MAX_TRACK = 16              # past this, decode/IO dominate tracking
_GB = 1 << 30


def avail_bytes() -> int:
    """Free physical memory (bytes); a conservative 8 GB if unknown."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(f, ctypes.c_ulonglong) for f in (
                               "total", "avail", "tpage", "apage", "tvirt",
                               "avirt", "aext")]
            m = _MS()
            m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.avail)
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:                                           # noqa: BLE001
        return 8 * _GB


def _override(workers: int | None) -> int | None:
    if workers:
        return max(1, int(workers))
    env = os.environ.get("REMOVE_SATELLITES_WORKERS", "").strip()
    return max(1, int(env)) if env.isdigit() else None


@dataclass(frozen=True)
class Plan:
    cores: int
    free_bytes: int
    track_workers: int
    export_workers: int
    export_threads: int         # threads inside each export process
    preview_threads: int
    cache_bytes: int            # GUI decoded-frame cache
    overridden: bool

    def describe(self) -> str:
        src = "override" if self.overridden else "auto"
        return (f"{self.cores} CPUs, {self.free_bytes / _GB:.0f} GB free ({src}): "
                f"tracking {self.track_workers} processes, export "
                f"{self.export_workers}×{self.export_threads} threads, preview "
                f"{self.preview_threads} threads, "
                f"{self.cache_bytes / _GB:.1f} GB frame cache")


def plan(width: int = 3840, height: int = 2160, radius: int = 10, *,
         workers: int | None = None) -> Plan:
    cores = os.cpu_count() or 1
    free = avail_bytes()
    budget = free * 0.6                          # leave the rest to the OS/UI
    frame = width * height * 3
    # median window ~4x over (buffer, warped copies, stack, strips) plus the
    # encoder's lookahead (~20 frames): ~2.9 GB per process at 4K, r=10
    per_export = (2 * radius + 1) * frame * 4 + frame * 20 + 256 * (1 << 20)
    per_track = frame * 6 + 128 * (1 << 20)
    by_mem_export = max(1, int(budget // per_export))
    by_mem_track = max(1, int(budget // per_track))

    fixed = _override(workers)
    if fixed:
        track_w = min(fixed, by_mem_track)
        export_w = min(fixed, by_mem_export)
    else:
        track_w = max(1, min(cores, MAX_TRACK, by_mem_track))
        export_w = max(1, min(cores // EXPORT_THREADS, by_mem_export))
    return Plan(cores=cores, free_bytes=free, track_workers=track_w,
                export_workers=export_w,
                export_threads=max(1, cores // export_w),
                preview_threads=cores,
                cache_bytes=int(min(4 * _GB, free // 4)),
                overridden=bool(fixed))
