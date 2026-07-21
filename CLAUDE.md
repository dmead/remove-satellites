# remove-satellites — trail remover for rotating star-field timelapses

Removes satellite/plane trails by rotation-compensated temporal median:
detect the sky's rotation (pole + rate), derotate so stars are static, median
out the transient streaks, re-rotate, mux the original audio back.

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

- **Rotation model**: `angle(k) = omega * (k - ref_index)` radians, linear in
  frame index; `RotationModel` is immutable. `rotation.estimate` fits it from
  star centroids via `cv2.estimateAffinePartial2D` (RANSAC) over several
  baselines; pole = fixed point of each transform, rate = weighted LS slope.
- **Pipeline aligns to the clip middle** (`Cleaner.ref_index = n//2`) via
  `Cleaner._angle_deg`, *not* to `model.ref_index` — this halves the max
  derotation and therefore the canvas padding (`_auto_pad`). Keep derotate and
  re-rotate using the *same* `_angle_deg` so they invert exactly.
- **Sign convention is calibrated, not assumed**: `_calibrate_sign` rotates the
  farthest frame's star points both ways and keeps the one that better matches
  the reference. Don't hard-code a sign.
- **All warps use `cv2.getRotationMatrix2D` + `warpAffine`** on a canvas padded
  by `pad` so no corner is clipped; output is re-rotated then cropped back to
  the source size. Points and images must use the *same* matrix convention.
- **Median is the hot loop**: `_median_u8` splits rows across `_NTHREADS`
  threads (numpy partition releases the GIL) and takes the middle rank. Don't
  replace with a single-threaded `np.median` — it's ~5× slower here.
- **Output is written through ffmpeg** (`video.FrameWriter`), not OpenCV's
  VideoWriter, so the source audio can be `-c:a copy`'d. Frames are fed as raw
  bgr24 on stdin.

## Layout

```
remove_satellites/core/video.py     probe / read_frame / iter_frames / FrameWriter
remove_satellites/core/stars.py     detect() -> [x,y,flux] intensity-weighted centroids
remove_satellites/core/rotation.py  estimate() -> RotationModel (pole, omega, residual)
remove_satellites/core/pipeline.py  Cleaner: .frame(i) preview, .run(out) full export
remove_satellites/cli.py            typer: detect / clean / gui
remove_satellites/gui/              app, window, preview (ndarray->QPixmap), worker (QThreads)
tests/synth.py                      synthetic rotating field + moving streak
```

## Data policy

No real captures live in this repo — tests synthesise a rotating field. The
`.gitignore` blocks common media/data extensions so a capture can't be
committed by accident.
