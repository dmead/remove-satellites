# remove-satellites

Remove satellite and aircraft trails from **rotating star-field timelapses**.

A fixed-tripod night-sky timelapse shows the stars wheeling about the
celestial pole. Trails (satellites, planes, meteors) are transient streaks
that cut *across* that rotation. remove-satellites:

1. **Tracks** the sky's motion frame to frame — stars followed with optical
   flow, a RANSAC homography per step — so any pole position works, including
   far off-frame (east/west/south views) and wide-angle lenses.
2. **Finds the foreground** (land, trees): pixels that stay put while the
   stars move.
3. For each frame, **warps its neighbours onto it** so the stars stand still
   and runs a **temporal median** that rejects the transient streaks while
   leaving the stars untouched. The foreground gets a plain median, so it
   isn't smeared.
4. Writes the result, copying the source audio.

Because the stars are held still during the median, the window can be as wide
as needed to kill a trail without ever smearing a star — the thing a naive
temporal median can't do on a rotating field.

## Install

Uses [uv](https://docs.astral.sh/uv/). ffmpeg must be on `PATH`.

```sh
uv sync --extra gui        # core + desktop GUI
uv run remove-satellites --help
```

## CLI

```sh
# track the sky motion and summarise it (cached as night.track.npz)
uv run remove-satellites detect "night.mp4"

# clean a clip (tracks the sky, finds the foreground; writes night_no-satellites.mp4)
uv run remove-satellites clean "night.mp4" -r 10

# manual override if tracking struggles (pole x,y and rad/frame)
uv run remove-satellites clean "night.mp4" --cx 1633 --cy 763 --rate -0.000326
```

Key options: `-r/--radius` temporal-median half-window (bigger = stronger trail
rejection), `--crf` x264 quality, `--no-foreground` to skip the land/tree mask,
`-j/--workers` process count.

**Using the machine.** Tracking and export run in parallel processes, sized
automatically from the CPU count, free RAM and the clip's frame size; the
plan is printed at the start (e.g. `32 CPUs, 127 GB free (auto): tracking 16
processes, export 16×2 threads, …`). Override with `-j N` or
`REMOVE_SATELLITES_WORKERS=N`. On a 32-thread i9-14900K a 4K, 763-frame clip
tracks in ~35 s (was ~7 min) and exports ~2.75× faster than one process.

## GUI

```sh
uv run remove-satellites gui
```

Open a clip and **Auto-detect sky motion** (the same engine as the CLI; it
tracks every frame the first time — minutes for 4K — then loads instantly
from the `.track.npz` cache). **Show foreground mask** tints the land/trees
that get a plain median. Scrub and toggle **Before/After**, pick a median
window, then **Export**. The **Manual pole override** group (pole x/y and
total spin) replaces the tracked motion when a clip defeats the tracker.

## How well it works

On a 449-frame 1080p clip, auto-detection located the pole to a **0.3 px**
fit residual and the full clean ran in ~85 s (32-thread median). Trails vanish;
stars stay pin-sharp because they are motionless during the median step.

## Layout

```
remove_satellites/core/      video (OpenCV read, ffmpeg write/concat/mux, frame cache),
                             stars (centroids), track (per-step homographies),
                             foreground (land/tree mask), pipeline (window-aligned
                             median, parallel export), hw (worker plan),
                             rotation (legacy pole+rate fit)
remove_satellites/gui/       PySide6 app: window, preview, background workers
remove_satellites/cli.py     typer CLI (detect / clean / gui)
tests/                       synthetic fixtures (no real capture needed)
```

## Data

No captures are committed to this repo — the tests generate a synthetic
rotating field on the fly. Video/image files are gitignored by default.

