"""remove-satellites — `clean`, `detect`, and `gui` commands."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _default_out(inp: Path) -> Path:
    return inp.with_name(inp.stem + "_no-satellites.mp4")


def _load_or_track(input: Path, track_path: Path | None):
    """Sky-motion track, cached next to the clip (tracking 4K is slow)."""
    from .core import track

    cache = track_path or track.default_cache(input)
    typer.echo(f"using cached track {cache}" if cache.exists()
               else "tracking sky motion...")
    return track.load_or_estimate(input, cache)[0]


def _describe(tr, w: int, h: int) -> None:
    from .core import track

    typer.echo("  " + track.summary(tr, w, h))


_WORKERS_HELP = ("processes for tracking/export (default: sized to this "
                 "machine's cores and free RAM)")


def _use_workers(workers: int | None, info, radius: int = 10) -> None:
    """Apply --workers everywhere downstream and say what will be used."""
    import os

    from .core import hw

    if workers:
        os.environ["REMOVE_SATELLITES_WORKERS"] = str(workers)
    typer.echo(hw.plan(info.width, info.height, radius).describe())


@app.command()
def detect(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    track_path: Path = typer.Option(None, "--track",
                                    help="track cache (.npz)"),
    workers: int = typer.Option(None, "--workers", "-j", help=_WORKERS_HELP),
):
    """Track the sky motion of a clip and summarise it."""
    from .core import video

    info = video.probe(input)
    _use_workers(workers, info)
    _describe(_load_or_track(input, track_path), info.width, info.height)


@app.command()
def clean(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(None, "--output", "-o", help="output .mp4"),
    radius: int = typer.Option(10, "--radius", "-r",
                               help="temporal median half-window (frames)"),
    crf: int = typer.Option(16, help="x264 quality (lower = better)"),
    track_path: Path = typer.Option(None, "--track",
                                    help="track cache (.npz)"),
    foreground: bool = typer.Option(True, help="detect and protect "
                                    "camera-fixed foreground (land, trees)"),
    ref: int = typer.Option(0, help="reference frame for manual override"),
    cx: float = typer.Option(None, help="override pole x (skips tracking)"),
    cy: float = typer.Option(None, help="override pole y"),
    rate: float = typer.Option(None, help="override rate (rad/frame)"),
    workers: int = typer.Option(None, "--workers", "-j", help=_WORKERS_HELP),
):
    """Remove satellite/plane trails from a star-field timelapse."""
    from .core import foreground as fgmod
    from .core import pipeline, rotation, track, video

    output = output or _default_out(input)
    info = video.probe(input)
    _use_workers(workers, info, radius)

    if cx is not None and cy is not None and rate is not None:
        model = rotation.RotationModel(
            center=(cx, cy), omega=rate, ref_index=ref,
            n_frames=info.frame_count, residual_px=float("nan"), n_pairs=0)
        sign = pipeline._calibrate_sign(input, model, info.frame_count // 2)
        tr = track.Track.from_rotation(model, sign)
        typer.echo("using manual rotation override")
    else:
        tr = _load_or_track(input, track_path)
        _describe(tr, info.width, info.height)

    mask = None
    if foreground:
        typer.echo("finding foreground...")
        mask = fgmod.static_mask(input, tr)
        typer.echo(f"  {100 * mask.mean():.1f}% of the frame is foreground")

    cleaner = pipeline.Cleaner(input, tr, radius=radius, fg_mask=mask)
    n = cleaner.info.frame_count

    with typer.progressbar(length=n, label="cleaning") as bar:
        done = {"k": 0}

        def prog(k, _n):
            bar.update(k - done["k"])
            done["k"] = k

        cleaner.run(output, crf=crf, progress=prog)

    typer.echo(f"wrote {output}")


@app.command()
def gui(output: Path = typer.Option(None, help="preset output path")):
    """Launch the desktop app."""
    import sys

    from .gui.app import run

    sys.exit(run(str(output) if output else None))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
