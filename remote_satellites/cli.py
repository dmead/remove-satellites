"""remote-satellites — `clean`, `detect`, and `gui` commands."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _default_out(inp: Path) -> Path:
    return inp.with_name(inp.stem + "_no-satellites.mp4")


@app.command()
def detect(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    ref: int = typer.Option(0, help="reference frame index"),
    samples: int = typer.Option(25, help="baselines sampled across the clip"),
):
    """Print the auto-detected sky rotation (pole + rate) for a clip."""
    from .core import rotation

    m = rotation.estimate(input, ref_index=ref, samples=samples)
    typer.echo(f"center      : ({m.center[0]:.1f}, {m.center[1]:.1f}) px")
    typer.echo(f"rate        : {m.omega:.6f} rad/frame")
    typer.echo(f"total       : {m.total_deg:.2f} deg over {m.n_frames} frames")
    typer.echo(f"residual    : {m.residual_px:.2f} px  ({m.n_pairs} baselines)")


@app.command()
def clean(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(None, "--output", "-o", help="output .mp4"),
    radius: int = typer.Option(10, "--radius", "-r",
                               help="temporal median half-window (frames)"),
    crf: int = typer.Option(16, help="x264 quality (lower = better)"),
    ref: int = typer.Option(0, help="reference frame for rotation fit"),
    samples: int = typer.Option(25, help="baselines sampled across the clip"),
    cx: float = typer.Option(None, help="override pole x (skips detection)"),
    cy: float = typer.Option(None, help="override pole y"),
    rate: float = typer.Option(None, help="override rate (rad/frame)"),
):
    """Remove satellite/plane trails from a rotating star-field timelapse."""
    from .core import pipeline, rotation, video

    output = output or _default_out(input)

    if cx is not None and cy is not None and rate is not None:
        info = video.probe(input)
        model = rotation.RotationModel(
            center=(cx, cy), omega=rate, ref_index=ref,
            n_frames=info.frame_count, residual_px=float("nan"), n_pairs=0)
        typer.echo("using manual rotation override")
    else:
        typer.echo("detecting sky rotation...")
        model = rotation.estimate(input, ref_index=ref, samples=samples)
        typer.echo(f"  pole ({model.center[0]:.0f},{model.center[1]:.0f}), "
                   f"{model.total_deg:.2f} deg, residual {model.residual_px:.2f}px")

    cleaner = pipeline.Cleaner(input, model, radius=radius)
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
