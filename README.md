# remote-satellites

Remove satellite and aircraft trails from **rotating star-field timelapses**.

A fixed-tripod night-sky timelapse shows the stars wheeling about the
celestial pole. Trails (satellites, planes, meteors) are transient streaks
that cut *across* that rotation. remote-satellites:

1. **Measures** the rotation — pole location and angular rate — by matching
   star centroids across frames and fitting a rigid model (RANSAC).
2. **Unwinds** it so the stars stand still.
3. Runs a **temporal median** that rejects the transient streaks while leaving
   the (now stationary) stars untouched.
4. **Re-applies** the rotation and writes the result, copying the source audio.

Because the stars are held still during the median, the window can be as wide
as needed to kill a trail without ever smearing a star — the thing a naive
temporal median can't do on a rotating field.

## Install

Uses [uv](https://docs.astral.sh/uv/). ffmpeg must be on `PATH`.

```sh
uv sync --extra gui        # core + desktop GUI
uv run remote-satellites --help
```

## CLI

```sh
# inspect the detected rotation
uv run remote-satellites detect "night.mp4"

# clean a clip (auto-detects rotation; writes night_no-satellites.mp4)
uv run remote-satellites clean "night.mp4" -r 10

# manual override if auto-detect struggles (pole x,y and rad/frame)
uv run remote-satellites clean "night.mp4" --cx 1633 --cy 763 --rate -0.000326
```

Key options: `-r/--radius` temporal-median half-window (bigger = stronger trail
rejection), `--crf` x264 quality, `--samples` baselines used for the fit.

## GUI

```sh
uv run remote-satellites gui
```

Open a clip, **Auto-detect rotation** (or set the pole and total spin by hand),
scrub frames and toggle **Before/After**, pick a median window, then **Export**.

## How well it works

On a 449-frame 1080p clip, auto-detection located the pole to a **0.3 px**
fit residual and the full clean ran in ~85 s (32-thread median). Trails vanish;
stars stay pin-sharp because they are motionless during the median step.

## Layout

```
remote_satellites/core/  video (OpenCV read, ffmpeg write+mux), stars (centroids),
                         rotation (pole+rate fit), pipeline (derotate·median·re-rotate)
remote_satellites/gui/   PySide6 app: window, preview, background workers
remote_satellites/cli.py typer CLI (detect / clean / gui)
tests/                   synthetic rotating-field fixtures (no real capture needed)
```

## Data

No captures are committed to this repo — the tests generate a synthetic
rotating field on the fly. Video/image files are gitignored by default.
