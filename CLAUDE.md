# remove-satellites — trail remover for star-field timelapses

Removes satellite/plane trails by star-aligned temporal median: track the
sky's frame-to-frame motion (homographies), warp each median window's
neighbours onto its centre frame so stars are static, median out the
transient streaks, give camera-fixed foreground a plain median, mux the
original audio back.

Parked ideas (trail annotation) live in
`docs/ROADMAP.md`.

## Environment

- **No system Python.** `python` on PATH is the WindowsApps stub. Everything
  runs through uv: `"$LOCALAPPDATA/Microsoft/WinGet/Links/uv.exe"`.
- **Before any uv command**: `. scripts/py.sh` (or export
  `UV_CACHE_DIR=D:/Temp/uv-cache` and `UV_PYTHON_INSTALL_DIR=D:/uv-pythons`).
  C: is nearly full; uv's defaults land on C: and must not.
- `uv sync --extra gui` to include PySide6. Tests: `uv run pytest -q`
  (synthetic only — `tests/synth.py`, no real captures needed).
- ffmpeg must be on PATH (chocolatey). Scratch/temp: `D:\Temp\...`, never C:.

## Design invariants (locked by tests — change test + code together)

- **Star detection subtracts a local background** (`stars.background`, a
  median at 1/8 scale) before thresholding. A global threshold fails on any
  frame with horizon glow (mono_east: threshold 323 on 8-bit data).
- **Sky motion is a chain of homographies** (`track.Track`, `steps[k]`: frame
  k → k+1), not a pole + rate. It covers far-off-frame poles (pure pan) and
  wide-angle projection, where the apparent rotation/scale drift over a clip.
  `Track.between(src, dst)` composes steps; backwards it must undo step
  `src-1` first (order matters — homographies don't commute).
- **Tracking fits the movers**: `_step` drops points that moved < 0.3 px when
  enough moved, so a static foreground can't win RANSAC with an identity fit.
  `finalize` rejects steps whose motion is > 25% off the clip median
  (washed-out frames lock onto the foreground) and gap-fills them.
- **Each median window is aligned to its own centre frame** — no global
  derotation, canvas padding or re-rotation. Only ≤ 2r chained steps are
  ever composed, so chaining drift stays sub-pixel.
- **Foreground mask** (`foreground.static_mask`): per-pixel vote of raw vs
  sky-aligned difference on background-subtracted frames, closed, then
  everything unreachable from the top edge through sky is foreground.
  Foreground gets a plain median; aligned samples landing on foreground or
  off-frame are excluded from the sky median by filling them alternately with
  0/255 (balanced fills leave the middle rank on the valid samples).
- **Sign convention is calibrated, not assumed** for the legacy pole + rate
  model (`_calibrate_sign`, then `Track.from_rotation`). Don't hard-code it.
- **Median is the hot loop**: `_median_u8` splits rows across `_NTHREADS`
  threads (numpy partition releases the GIL) and takes the middle rank. Don't
  replace with a single-threaded `np.median` — it's ~5× slower here.
- **Output is written through ffmpeg** (`video.FrameWriter`), not OpenCV's
  VideoWriter, so the source audio can be `-c:a copy`'d. Frames are fed as raw
  bgr24 on stdin.

## Layout

```
remove_satellites/core/video.py       probe / read_frame / iter_frames / FrameWriter
remove_satellites/core/stars.py       background() / detect() -> [x,y,flux] centroids
remove_satellites/core/track.py       estimate() -> Track (per-step homographies)
remove_satellites/core/foreground.py  static_mask() -> camera-fixed pixels
remove_satellites/core/rotation.py    legacy pole + rate fit (GUI manual controls)
remove_satellites/core/pipeline.py    Cleaner: .frame(i) preview, .run(out) full export
remove_satellites/cli.py              typer: detect / clean / gui (track cached as <clip>.track.npz)
remove_satellites/gui/                app, window, preview, worker — still pole + rate based
tests/synth.py                        synthetic field: rotation, drift (pan), ground, streak
```

## Data policy

No real captures live in this repo — tests synthesise a rotating field. The
`.gitignore` blocks common media/data extensions so a capture can't be
committed by accident.
