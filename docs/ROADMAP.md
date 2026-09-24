# Roadmap / deferred work

Parked ideas and known limitations, captured so they aren't lost.

## 1. Off-frame pole robustness — DONE (2026-09)

Implemented as `core/track.py`: per-step homographies instead of pole + rate,
with each median window aligned to its own centre frame. Prompted by
`mono_east` (east-facing, 4K, wide-angle), whose apparent per-frame rotation
and scale drifted across the clip — not even a similarity. Kept below for
history.

**Behaviour before the fix.** `rotation.estimate` fits a *pure rotation about a pole*
(a single `center` + `omega`). This works when the pole is in the frame **or**
off-frame, as long as the star trails are still visibly curved.

Verified on synthetic fields (pole placed increasingly far outside the frame):

| pole location            | fit residual | trail removed | stars kept |
|--------------------------|-------------:|:-------------:|:----------:|
| in frame                 | 0.47 px      | yes           | yes        |
| just outside (~80 px)    | 0.47 px      | yes           | yes        |
| far outside (~1500 px)   | 0.44 px      | yes           | yes        |
| very far (~10 FOV widths)| 0.50 px      | yes           | **no**     |

**The failure mode.** When the pole is so far away that the trails are
essentially straight and parallel (a field pointed near the **celestial
equator** — common for E/W/S nightscapes), the motion is indistinguishable
from a pure pan. The pole solver can't localize the centre, falls back to the
frame centre, and derotating about the wrong centre fails to cancel the
motion → stars get smeared/erased along with the trails.

It also fails **silently**: `residual_px` reflects the per-baseline affine fit,
not the final `pole + rate` reconstruction, so it stays low even when the
reconstruction is wrong.

**Fix plan.** Align each frame by its *full fitted rigid transform*
(rotation **and** translation) relative to a reference — smoothed across the
clip — instead of reconstructing motion from a single pole + rate. This covers
any pole position including "at infinity" (pure pan). `pole + rate` becomes a
reported summary, and the alignment residual becomes an honest end-to-end
quality metric.

## 2. Satellite trail annotation

The removed content (`original − cleaned`) already isolates every streak.
Connected-component + track each one across frames to get its time range,
pixel path, direction, and angular rate. Outputs:
- a labelled overlay burned into the video ("Sat 1", t=3.2–4.0 s, …), and
- a CSV catalogue of all trails.

Proposed surface: `remove-satellites annotate INPUT`.

