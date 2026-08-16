# Phase 1 — runtime: more runs per GPU-hour

**Date:** 2026-08-16 · **Phase:** 1 of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 89, 90, 91, 92, 95

## Goal

Cut the wall clock of a federation round by at least **2×** at identical image-visits,
without moving the holdout mAP outside the run-to-run spread. The last full run held the
GPU at **27 % mean utilisation** with **5 087 MiB of 16 303** peak VRAM: the card is
mostly idle and mostly empty, and every later phase is paid for in runs.

Speed that costs accuracy is not speed, it is a shorter run. Every change here is
accepted only against an unchanged holdout curve and a still-moving aggregate checksum.

## Hard constraints

- **Measure before changing.** Land the profiling deliverable first and commit its
  output. A lever justified by a guess is a lever that gets reverted.
- **`pipeline/` must not modify `my-project/`.** Levers that live in `client_app.py` or
  `pyproject.toml` go on their own branch with their own prompt (marked ⚠ below);
  do not smuggle them into a pipeline commit.
- **One lever per commit**, each with its own before/after timing on the same fleet
  fingerprint. Two levers in one commit means neither is measured.
- **Do not touch** `torch.compile`, multi-GPU, or mixed-resolution training. Compile's
  warmup is paid per short-lived client process; there is one card; mixed resolution
  changes the very number being held constant.
- No new dependency. Everything here is a keyword argument, a config value, or `time.perf_counter`.

## Inputs — verified facts, not assumptions

Checked against ultralytics **8.4.115** in `C:\Users\PRANAS\venvs\fl_yolov8`:

| Fact | Where |
|---|---|
| `workers=0 if IS_WINDOWS else 8` — decode runs on the training thread | `my-project/my_project/client_app.py` ~L223 |
| `cache` defaults to `False`; images are re-decoded from JPEG every epoch | ultralytics defaults |
| `options.backend.client-resources.num-gpus = 1.0` serialises clients | `my-project/pyproject.toml` L73 |
| A fresh `YOLO` is built from yaml and reloaded every round | `client_app.py`, `task.py` |
| Reference run: 3 296 s, 82.2 Wh, 27 % util, 5.1 GB peak, 6×4×6×1 400 | `STATUS.md` |

## Deliverables

| # | File | What it does |
|---|---|---|
| 1 | `pipeline/profile.py` | parses a run's logs into a per-phase seconds breakdown: shard scan, model construction, AMP check, warmup, steady-state, validation, checkpoint write, serialisation, server aggregation, **idle**. Written to `pipeline/.state/profile-<run>.json` and printed as a table |
| 2 | `pipeline/gpu.py` (extend) | sample utilisation and VRAM at a fixed interval during a run, so "27 %" becomes a curve rather than a mean |
| 3 | ⚠ `my-project/pyproject.toml` | `client-resources.num-gpus` driven by a run-config value, defaulting to today's `1.0`. The pipeline passes it |
| 4 | ⚠ `client_app.py` | `cache` and `workers` as run-config values; `cache="ram"` the demo-profile default, `False` at full scale where RAM will not hold the shard |
| 5 | `pipeline/build_fleet.py` (assert) | a test that a fleet rebuild is what invalidates `labels.cache`, and that a federation across rounds does not — turning lever 4 from hope into a guarantee |
| 6 | `docs/RUNBOOK.md` (section) | the concurrency/VRAM table: per-vehicle image count → measured peak VRAM → how many clients fit |

Deliverables 3 and 4 are ⚠ and belong on a separate branch from 1, 2, 5, 6.

## The levers, in the order to try them

1. **`num-gpus = 0.33`** — three clients share the card. Mathematically a no-op: clients
   are independent within a round. Expect ~2–2.5×, and expect it to be the whole result.
2. **`cache="ram"`** — 1 400 images at 640 px is ~2–3 GB. Removes JPEG decode from the
   step loop, the prime suspect behind 27 % utilisation with `workers=0`.
3. **Windows dataloader** — leave `workers=0` if lever 2 lands; the existing comment is
   right that spawned workers deadlock inside a Ray actor. Do not "fix" that comment by
   experiment on the full profile.
4. **Label-cache reuse across rounds** — 6 vehicles × 6 rounds is 36 scans of the same
   directory.
5. **Persistent client actors** — keep the constructed `YOLO` in Flower node state.
   Small, free, and it removes a per-round fixed cost that grows with round count.

## Definition of done

```bash
python -m pytest pipeline/tests -q                    # 59 tests, exit 0
python -m pytest my-project/tests -q                  # 31 tests, exit 0
python -m pipeline.profile --last                     # prints the breakdown table
python -m pipeline.runner --all --yes                 # demo profile, before and after
python -m pipeline.holdout --evaluate                 # mAP unchanged within spread
python -m pipeline.verify                             # four criteria green
```

Recorded in the commit body, with real numbers:

- seconds per 1 000 image-visits, before and after, same fleet fingerprint
- mean and peak GPU utilisation, before and after
- holdout mAP50 and mAP50-95 before and after, with the phase-3 spread quoted
- the aggregate checksum sequence, proving it still moves every round

## Out of scope

Model architecture, batch size heuristics beyond the existing
`get_optimal_batch_size()` guard, the LR schedule (that is phase 2), anything that
changes what is learned rather than how fast it is learned, and any UI work.
