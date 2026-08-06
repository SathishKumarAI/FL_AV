# federated-yolov8 — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-08-06

- **Where I stopped:** the federation has a **scale** for the first time. A shared
  holdout exists, the global model is scored on it out of band, and a centralised
  baseline trains on the pooled data for the same budget. Five increments landed on
  `feat/pipeline-observability`: the premium dashboard, Dirichlet partitioning, the
  holdout + baseline, the strategy registry, and one real bug fix.

## The result of the long run (6 rounds × 4 local epochs × 6 vehicles × 1 400 images)

Finished 2026-08-06 01:51, 3 296 s of GPU time, 82.2 Wh, peak VRAM 5 087 MiB of
16 303 (31 %), mean utilisation 27 %.

| measured on | round 1 | → | round 6 |
|---|---|---|---|
| **shared holdout, 1 000 images no vehicle saw** | 0.3543 | | **0.4334** mAP50 (0.2454 mAP50-95) |
| clients' own val splits (the old number) | — | | 0.4642 mAP50 |

Aggregate checksum moved every round: 159.9 → −158.7 → −415.8 → −583.8 → −759.5 →
−823.5. All four pass criteria green during the run.

The holdout curve is the honest one, and it is **0.031 lower** than the
self-evaluated number — which is the size of the flattery the old metric contained.
Previous session's run reached 0.320 self-evaluated on 2 effective epochs; this one
did 24 effective epochs on 4.7× the data.

Per vehicle, best to worst on their own splits: v1 daytime city 0.4754, v9 daytime
city 0.4513, v8 parking/tunnel 0.4324, v10 night 0.4243, v2 night 0.4163, v5
dawn/dusk 0.4069. Spread is the point — those are different distributions, which is
exactly why the holdout had to exist.

- **Next action:** read `pipeline/.state/baseline.json` — a centralised model was
  training on the pooled 8 400 images for 24 epochs (the same image-visits the fleet
  made) when this was written. `python -m pipeline.baseline` prints the gap and what
  fraction of the ceiling the federation retains. That number is the actual result of
  this project so far; everything before it was a number without a scale.

  Then: backlog 42 (seeds — one run is an anecdote), 31 (the rounds × epochs sweep at
  constant product), 27 and 30 (freeze the backbone for round 1, LR schedule for
  short rounds — both ⚠, both in `docs/ML_PLAN.md`).

## What exists now that did not before

| | Command |
|---|---|
| Shared holdout, carved before the fleet so no vehicle can see it | `python -m pipeline.holdout --build --size 1000` |
| Global model scored on it, per round | `python -m pipeline.holdout --evaluate` |
| Centralised ceiling on pooled data at a matched budget | `python -m pipeline.baseline --rounds 6 --local-epochs 4` |
| Dirichlet partitioning, α as the skew knob | `... --partition dirichlet --alpha 0.3` |
| Any of 12 Flower strategies | `... --strategy fedadam` |
| The dashboard, rebuilt | `python -m pipeline.server` |

Stage chain is now: env → dataset → populate → **holdout** → fleet → sanity →
federate → **evaluate** → verify → **baseline** (gated).

## Traps confirmed again this session

- **Anything constructed with `cwd=my-project` has side effects on relative paths.**
  A strategy-registry probe truncated `logs/metrics.csv` during the final minute of
  the six-round run, and the run's own verify then failed for want of rows it had
  written. Fixed in `utils/metrics_logger.py` (the file starts on the first row, not
  on construction) — but the general lesson stands: run throwaway checks from the
  repo root or under pytest's `_isolate_cwd`.
- **`build_fleet` rmtree's `pipeline/vehicles/batch/`.** Never run it while a
  federation is in flight. The Dirichlet work was verified against the real attribute
  index without materialising anything, for exactly this reason.
- Peak VRAM at 1 400 images per vehicle is 5.1 GB, not the 15.9 GB the 6 308-image
  shard needs. There is room to pack clients concurrently at this profile (backlog 89).

- **Environment (the part that costs an hour if you forget it):** use the venv at
  `C:\Users\PRANAS\venvs\fl_yolov8`, built on python.org 3.12 — *not* conda. Smart App
  Control blocks conda-forge's `_bz2.pyd`; see [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md).
  Export `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or flwr
  builds its own runtime env with the CPU-only torch wheel and every client trains on
  CPU at 5.5× the wall clock with no error anywhere.

- **Data: done.** All ten shards hold real BDD100K, hardlinked onto the kagglehub
  cache. The attribute index (79 863 images) is cached at
  `pipeline/.state/attributes.json`. Full instructions in [`docs/DATASET.md`](docs/DATASET.md).
