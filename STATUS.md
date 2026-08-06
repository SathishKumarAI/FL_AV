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

**One caveat, stated rather than buried.** The fleet on disk was built *before* the
holdout existed, and `python -m pipeline.validate` reports that 439 of the 1 000
held-out images sit in vehicles' **val** splits. No client trained on them — train and
val pools are disjoint by construction — so the global model never saw them and the
curve above is sound. But those 439 did feed clients' self-evaluation, so the
0.4642 self-reported number is the affected one. Rebuilding the fleet clears it, and
the fleet check now forces that rebuild.

Previous session's run reached 0.320 self-evaluated on 2 effective epochs; this one
did 24 effective epochs on 4.7× the data.

Per vehicle, best to worst on their own splits: v1 daytime city 0.4754, v9 daytime
city 0.4513, v8 parking/tunnel 0.4324, v10 night 0.4243, v2 night 0.4163, v5
dawn/dusk 0.4069. Spread is the point — those are different distributions, which is
exactly why the holdout had to exist.

## The centralised ceiling, and why the retention figure is a lower bound

`pipeline/.state/baseline-14000img-24ep.json`: **0.4771 mAP50 / 0.2659 mAP50-95** on
the same holdout. Against the federation's 0.4334 that is a gap of 0.0437, and the
federation retains **90.8%** of it.

Read that as a **lower bound**, not the result. The run pooled all ten materialised
shards, not the six that trained, so it saw 14 000 images for 24 epochs -- 336 000
image-visits against the federation's 201 600, a 1.667x advantage in data *and*
compute. `pooled_names()` now defaults to the shards that actually trained and the
CLI prints the parity ratio, so the next one is matched by construction.

**To get the matched number** (about 90 min, and it needs a fresh federation because
the fleet has since been rebuilt):

```powershell
.\scripts\run_pipeline.ps1 -Profile full -Vehicles 6 -PerVehicle 1400 -Rounds 6 -Epochs 4 -Baseline
```

## Running it standalone found seven defects that never showed interactively

Every one of these only appears when the pipeline is driven by a script rather than
typed into a shell, which is how anyone reproducing this project would run it.

| | Defect | Why it hid |
|---|---|---|
| 1 | `python -m ultralytics.cfg` stopped being executable in ultralytics 8.4 | the sanity stage had a marker and was being skipped |
| 2 | sanity trained on `data.runtime.yaml`, which a *client* writes at runtime | it only passed where a client had already run |
| 3 | the runner's output thread died on a cp1252-unencodable line | redirected stdout only |
| 4 | `flwr` resolved from PATH | an activated venv has it; a script's shell does not |
| 5 | flwr's own banner emoji killed it under cp1252 | same |
| 6 | flwr launches `flower-superlink` from PATH too | our children spawn children |
| 7 | the holdout scorer mixed this run's checkpoints with the last run's | the directory is never cleared |

An eighth, found while reading the output: the checksum criterion concatenated every
server log it could find, so a three-round run was judged on eleven checksums from
three runs.

- **Next action:** read `pipeline/.state/baseline.json` — a centralised model was
  training on the pooled 8 400 images for 24 epochs (the same image-visits the fleet
  made) when this was written. `python -m pipeline.baseline` prints the gap and what
  fraction of the ceiling the federation retains. That number is the actual result of
  this project so far; everything before it was a number without a scale.

  Then: backlog 42 (seeds — one run is an anecdote), 31 (the rounds × epochs sweep at
  constant product), 27 and 30 (freeze the backbone for round 1, LR schedule for
  short rounds — both ⚠, both in `docs/ML_PLAN.md`). Each is now one command:
  `python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes`.

  A first real comparison is already on disk
  (`pipeline/.state/experiments/20260806-034438.md`): fedavg vs fedadam at demo scale,
  same fleet fingerprint `a0b504089c0e`, 2 rounds × 1 epoch. fedavg 0.0042 holdout
  mAP50, fedadam 0.0000. Too small a budget to conclude anything about the
  strategies — at 2 rounds the server-side optimiser has not had time to help — but
  it demonstrates the machinery: one setting varied, identical data proven by the
  fingerprint, both scored on the same held-out images.

## What exists now that did not before

| | Command |
|---|---|
| Shared holdout, carved before the fleet so no vehicle can see it | `python -m pipeline.holdout --build --size 1000` |
| Global model scored on it, per round | `python -m pipeline.holdout --evaluate` |
| Centralised ceiling on pooled data at a matched budget | `python -m pipeline.baseline --rounds 6 --local-epochs 4` |
| Dirichlet partitioning, α as the skew knob | `... --partition dirichlet --alpha 0.3` |
| Any of 12 Flower strategies | `... --strategy fedadam` |
| The dashboard, rebuilt | `python -m pipeline.server` |
| Shard validation — six ways a fleet can be quietly wrong | `python -m pipeline.validate` |

Stage chain is now: env → dataset → populate → **holdout** → fleet → **validate** →
sanity → federate → **evaluate** → verify → **baseline** (gated).

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
