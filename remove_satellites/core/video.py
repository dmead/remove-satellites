"""Video IO: read frames with OpenCV, encode + mux audio with ffmpeg.

We read with OpenCV (simple, random access for previews) but *write* through
ffmpeg: OpenCV's VideoWriter can't copy the source audio track, and ffmpeg
gives us H.264 quality control + `-c:a copy` for free. Frames are fed to
ffmpeg's stdin as raw BGR24 (the native OpenCV byte order).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 24.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """Read a single BGR frame by index (for previews)."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"frame {index} out of range in {path}")
        return frame
    finally:
        cap.release()


def iter_frames(path: str | Path, start: int = 0, stop: int | None = None):
    """Yield BGR frames [start, stop) in order."""
    cap = cv2.VideoCapture(str(path))
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while stop is None or i < stop:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            i += 1
    finally:
        cap.release()


def concat(segments, out_path, *, source=None) -> None:
    """Join same-encoded video segments losslessly (concat demuxer, stream
    copy) and mux the source's audio, if any, onto the result."""
    out_path = Path(out_path)
    lst = out_path.with_name(out_path.name + ".concat.txt")
    lst.write_text("".join(
        "file '{}'\n".format(str(Path(s).resolve()).replace("'", r"'\''"))
        for s in segments), encoding="utf-8")
    args = [ffmpeg_bin(), "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(lst)]
    if source is not None:
        args += ["-i", str(source), "-map", "0:v", "-map", "1:a?"]
    args += ["-c", "copy", "-movflags", "+faststart", str(out_path)]
    try:
        r = subprocess.run(args, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed: {r.stderr.decode(errors='replace')}")
    finally:
        lst.unlink(missing_ok=True)


class FrameCache:
    """Thread-safe LRU of decoded frames, bounded by bytes.

    Scrubbing the preview reuses most of the previous median window; keeping
    decoded frames avoids re-seeking (a random 4K seek costs ~0.4 s)."""

    def __init__(self, budget_bytes: int = 2 << 30):
        import threading
        from collections import OrderedDict
        self.budget = budget_bytes
        self._d: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, k: int):
        with self._lock:
            f = self._d.get(k)
            if f is not None:
                self._d.move_to_end(k)
            return f

    def put(self, k: int, frame: np.ndarray) -> None:
        with self._lock:
            if k in self._d:
                return
            self._d[k] = frame
            self._bytes += frame.nbytes
            while self._bytes > self.budget and len(self._d) > 1:
                _, old = self._d.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


class FrameWriter:
    """Encode BGR frames (fed via .write) to H.264, muxing audio from `source`.

    Usage:
        with FrameWriter(out, w, h, fps, source=src, crf=16) as w:
            for f in frames: w.write(f)
    """

    def __init__(self, out_path, width, height, fps, *, source=None,
                 crf=16, preset="medium", threads: int | None = None):
        self.proc: subprocess.Popen | None = None
        args = [
            ffmpeg_bin(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
        ]
        if source is not None:
            args += ["-i", str(source)]
        args += ["-map", "0:v"]
        if source is not None:
            args += ["-map", "1:a?", "-c:a", "copy"]
        args += [
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
        if threads:                       # several encoders share the CPU
            args += ["-threads", str(threads)]
        args.append(str(out_path))
        self._args = args

    def __enter__(self) -> "FrameWriter":
        self.proc = subprocess.Popen(
            self._args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.proc
        if self.proc.stdin:
            self.proc.stdin.close()
        err = self.proc.stderr.read() if self.proc.stderr else b""
        rc = self.proc.wait()
        if rc != 0 and exc_type is None:
            raise RuntimeError(f"ffmpeg failed ({rc}): {err.decode(errors='replace')}")
