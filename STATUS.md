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

## The result: federation against a budget-matched ceiling

Both sides made **201 600 image-visits** — 6 vehicles × 1 400 images × 6 rounds × 4
local epochs, against 8 400 pooled images × 24 epochs. Parity is asserted in the
artifact (`ratio 1.0, matched: true`), not claimed in prose.

| on 1 000 held-out images | federated | centralised | gap | retained |
|---|---|---|---|---|
| mAP50 | 0.4173 | **0.4936** | +0.0763 | **84.5 %** |
| mAP50-95 | 0.2313 | **0.2770** | +0.0457 | 83.5 % |

Federated learning on this fleet costs about **15 % of the achievable accuracy** at an
identical training budget, in exchange for never pooling the data. That is the result
this project existed to produce, and until today it could not have been stated: the
metric was a client scoring itself on its own conditions, and the first ceiling ran
with 1.667× the federation's budget.

The federated curve is monotonic across all six rounds — 0.3329, 0.3763, 0.3974,
0.4066, 0.4120, 0.4173 — and the aggregate checksum moved every round.

**One number that needs a caveat.** An earlier ceiling, trained on 14 000 images for
the same 24 epochs (336 000 visits, 1.667× the budget), scored **lower**: 0.4771
against this one's 0.4936. More data and more compute produced a worse model, which
means run-to-run variance here is at least ±0.016 — larger than several of the
differences this project might want to call results. Backlog 42 (seed repeats) is
therefore not optional; it is the prerequisite for believing any comparison, and the
Metrics tab already groups repeats and shows their spread.

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

## Next session — carry these forward

Ordered by what they unblock, not by size.

| # | Task | Why it matters | Size |
|---|---|---|---|
| 1 | ⚠ **Give `my-project`'s loggers absolute paths.** `utils/logging_setup.py` configures `logs/server.log` **relative to the CWD**, at import time. So merely importing `my_project.server_app` — which `pytest my-project/tests` does at collection, from the repo root — creates an empty `logs/server.<pid>.log`. That file then looked newer than the real federation's log and made `verify` report `need >=2 rounds to tell, saw 0` right after a six-round run had succeeded. The pipeline is now robust to it (`logparse.latest_run_log` only trusts a log that aggregated a round), but the cause is still there, and it also means any import scatters log files wherever you happen to be standing. Gated: own branch, own prompt. | small |
| 2 | **Backlog 30 — LR schedule for short rounds.** The Metrics tab now shows box, cls **and** dfl all *rising* across the four epochs of every round: each client ends the round worse than the aggregate it started from. Warmup is three epochs of a four-epoch round, so the schedule never leaves warmup. This is the most likely reason 24 effective epochs only reached 0.4173. ⚠ touches `client_app.py`. | medium |
| 3 | **Backlog 42 — repeats across seeds.** `python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes`. Until the spread across repeats is known, no difference between two approaches means anything. The Metrics tab already groups repeats and shows the spread; it just needs runs to group. | 3 × one run |
| 4 | **Backlog 31 — rounds × epochs at constant product.** 12×2, 6×4, 3×8 at the same image-visits. Directly measures client drift, and item 2 predicts the answer: fewer local epochs should win. | 3 runs |
| 5 | **Backlog 80 — MLflow.** It is wired and it refused this run: *"the filesystem tracking backend is in maintenance mode"*. It needs a SQLite backend (`mlflow-tracking-uri sqlite:///…`), so this is more than calling the sink. | small |
| 6 | **Backlog 36 — class imbalance.** `car` is 55.4% of all objects and `train` has 29 instances fleet-wide. Averaged mAP flatters a car detector; per-class numbers are the honest report. | medium |
| 7 | **Re-run the pre-holdout comparison.** Reports written before today mix runs in their stored `learning` block (the four provenance bugs). They are marked in the Metrics tab but cannot be repaired retroactively — only superseded by fresh runs. | free, with 3 |

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
