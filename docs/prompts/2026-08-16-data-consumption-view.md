# Prompt — show what YOLO consumes, not only how much of it

**Branch:** `feat/data-consumption-view` · **Written:** 2026-08-16

> Honesty note: this was written after the code, not before it. The session began from
> a broad request ("make the data and how the model uses it visible, more UI, more live
> feed") and the shape of the increment was only decided once the repo had been read.
> `docs/prompts/README.md` is right that a reconstructed prompt records what happened
> rather than what was intended; this one is kept because the *rejected* options below
> are the part worth keeping, and they were genuinely weighed before any file changed.

## The brief

The Data tab answers *how much*: images per shard, class histograms, weather mix,
label density, holdout leakage. It cannot answer *is the model looking at what we think
it is looking at* — the question that every silent failure in this project's history
turned out to hinge on. Nothing in the dashboard has ever drawn a label.

Make the data path visible:

1. Draw each shard frame's label file over the frame.
2. Surface what the trainer itself saw and predicted, per vehicle.
3. Do not add a service, a daemon, a port, or a dependency to do it.

## What was found first, and what it changed

Ultralytics already writes, per vehicle, per round, into its run directory:
`train_batch{0,1,2}.jpg` (real batches after mosaic and augmentation, boxes drawn),
`labels.jpg`, `val_batch{n}_labels.jpg` beside `val_batch{n}_pred.jpg`,
`confusion_matrix_normalized.png`, the PR/F1 curves, `results.png`. Fourteen files per
vehicle, on disk since the last run, never once looked at.

That inverted the task: the work is not *producing* these pictures, it is *serving*
them. Rule 4 — assemble before building.

## Rejected

| Option | Why not |
|---|---|
| **FiftyOne** | The right tool for the job it does, and it costs a MongoDB process, a second web app on a second port, and a ~400 MB install to show pictures this repo already has on disk. Its real capability — embedding-based near-duplicate and label-error search — is a phase-4 question, not a UI one. Kept as a candidate in `docs/OSS_TOOLING_REVIEW.md`, not wired here. |
| **`supervision`** | Would draw the boxes server-side. But then the frame goes over the wire twice, a rendering dependency enters the project, and the browser can no longer toggle the overlay without a refetch. The label format is already normalised; an SVG over the `<img>` needs no library at all. |
| **Rerun** | Live streaming visualisation, genuinely good, and it would mean logging from `client_app.py` — the ⚠ zone — for a view of data that is not time-series. |
| **A second dashboard page** | The Data tab is where the reader already is. |

## Built

| | |
|---|---|
| `pipeline/train_artifacts.py` | locates each vehicle's ultralytics run directory by **glob**, not by construction (the path depends on that machine's ultralytics `runs_dir` setting), and allowlists which filenames may be served |
| `dataset_stats.boxes()` | one frame's label rows, normalised, malformed rows skipped rather than raised on |
| 3 routes | `/api/shard-labels/<vid>/<name>`, `/api/train-artifacts`, `/api/train-artifact/<vid>/<name>` |
| `static/js/consumed.js` | the unit-square SVG overlay, and the gallery of the trainer's pictures grouped as *batches as fed* / *truth vs prediction* / *where it fails* |

## Verification

- `pytest pipeline/tests -q` → 121 passed.
- Three new tests: the allowlist refuses `weights/best.pt`, `args.yaml` and `../../../CLAUDE.md`;
  a label file with two broken rows still yields its two good ones; and the overlay
  geometry is executed under node against a fake DOM.
- The geometry test was mutated (`b.w / 2` → `b.w / 3`) and **failed**, then restored
  and passed — a check that cannot fail is not a check.
- Live against a running server: `labels.jpg` 200 image/jpeg 129 949 B,
  `confusion_matrix_normalized.png` 200 image/png 289 765 B, `weights/best.pt` 404,
  traversal 404. `/api/shard-labels/1/00091078-7cff8ea6.jpg` → 23 boxes, 0 outside `[0,1]`.
- **Not verified: the rendered page.** Both browser automation paths were unavailable
  in this session (extension not connected; the devtools profile was already in use).
  The geometry, the routes and the empty states are covered by the tests above; the
  visual result is not. Say so rather than imply otherwise.
