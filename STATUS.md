# federated-yolov8 — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-08-16

## Where I stopped

Phases 0 and 1 of [`docs/PHASED_PLAN.md`](docs/PHASED_PLAN.md) are **done and measured**,
and half of phase 2. Ten branches, all pushed, all with PRs open (#44–#53), **none
merged** — `main` is ~70 commits behind and that is now the largest single item
outstanding.

The headline: **runs are 1.94× faster on 43 % less energy, and the detector starts from
COCO instead of from noise.** The second of those turned out to be worth more than
everything else in the session put together.

## The head was random, and nobody had looked

`YOLO(yolov8s-13.yaml).load(yolov8s.pt)` prints `Transferred 349/355 items`. The six it
cannot transfer are the three classification convolutions — 80 classes against 13 — so
they were **randomly initialised in every run this project has ever done**.

Nine of BDD100K's thirteen classes are COCO classes. Copying those rows
(`warm_start_head` in `get_set_model.py`), measured on the 1 000-image holdout, same
fleet, 1 400 images/vehicle:

| head | untrained | round 1 | round 2 |
|---|---|---|---|
| random | 0.0053 | 0.1277 | 0.1690 |
| **warm-started** | **0.2582** | 0.1924 | 0.2073 |

**0.2582 before a single gradient step** — 59 % of what six rounds × four epochs
previously reached, for free.

**And the thing it exposed.** The warm-started model is better *untrained* than after
two rounds. Round 1 costs it 0.066 mAP50. With a random head there was nothing to
damage, so that cost had always been invisible.

## Phase 1: the card was empty, not slow

`pipeline/profile.py` (new) answers where a round's seconds go, from timestamps the logs
already carried — so the 3 296 s reference run was profiled after the fact:

**85.3 % train, 13.8 % evaluate, 0.6 % idle, 72 client episodes never overlapping.**

Orchestration was never the problem. Clients were serialised on a card they used a third
of. Four arms at 1 400 images/vehicle, 6 × 2 rounds × 1 epoch, one fleet:

| `--gpu-fraction` | `--cache` | wall | util | peak VRAM | energy |
|---|---|---|---|---|---|
| 1.0 | — | 562.1 s | 19.3 % | 6 453 MiB | 10.36 Wh |
| 1.0 | ram | 595.2 s | 22.7 % | 6 313 MiB | 10.67 Wh |
| 0.5 | — | 394.9 s | 29.7 % | 10 474 MiB | 8.37 Wh |
| **0.33** | — | **289.2 s** | 30.9 % | 15 468 MiB (94.9 %) | **5.92 Wh** |

```bash
python -m pipeline.runner --all --yes --gpu-fraction 0.33      # the new default to use
```

Three clients hold 94.9 % of the card, so 0.33 is the floor on this hardware.

## Three things that were tried and did not work

Written down because a lever that was tried and failed is more useful than one that
looks untried.

| | Result |
|---|---|
| `cache="ram"` | **5.9 % slower.** Utilisation with the whole shard in RAM still only reached 22.7 %, so JPEG decode was never what the card waited for. That also makes the dataloader-workers lever moot |
| Persistent client actors | **cut without running it.** Phase 0 caps all per-round fixed cost at 8.6 s of 3 266 s |
| One LR anneal across the run | **−0.0079 mAP50, negative at six of six rounds.** Inside the ±0.016 spread, so "no difference" — but not what a win looks like. Implemented and tested on `feat/one-lr-anneal-across-rounds`, **not for merge** |

## Four bugs, three found by running the thing

| | |
|---|---|
| `pipeline/mlflow_sink.py` **was never called by anything** | MLflow held ultralytics' training curves and no federation at all — no aggregate checksum, no energy. Wired into the runner |
| MLflow refused every write | mlflow 3.15 rejects the `file://` backend outright. Now `sqlite:///pipeline/mlruns/mlflow.db`, one experiment for both writers |
| The runner's pyproject restore ran `git checkout --` | which discards *uncommitted* edits too. It deleted a run-config key mid-session and the next run failed on a value that had been there minutes earlier. Snapshot-based now |
| `/api/run` adopted the config before validating it | a refused run still replaced the config the stage table previews |

## Next action

1. **Merge the stack.** `main` last moved at **PR #29**; twenty PRs are open behind it.
   Everything below is gated on this, and it is the one step nothing here could do —
   `gh pr merge` was refused by a permission classifier, so it needs a human or an
   allowlist entry.

   The branches are a **linear stack**, each PR based on its parent, so every diff is
   reviewable on its own and each merge collapses the next one's. Merge bottom-up:

   ```
   39 → 43 → 44 → 45 → 46 → 47 → 48 → 49 → 50 → 51 → 52
   ```

   | PR | What it lands |
   |---|---|
   | #39, #43 | the previous session's pipeline, dashboard, holdout and size skew |
   | **#44** | `pipeline/profile.py` — phase 0 |
   | **#45** | `/api/run` validates before adopting |
   | **#46**, **#48** | MLflow: sqlite backend, and a sink that is actually called |
   | **#47** | `--gpu-fraction` — the 1.94× |
   | **#49** | `--cache`, measured and rejected; phase 1 settled |
   | **#50** | the pyproject restore that deleted uncommitted edits |
   | **#51** | the COCO head warm start |
   | **#52** | this file |
   | ~~#53~~ | the LR anneal — **draft, do not merge.** Experiment record only |

   Also open and pre-dating this session: **#36** (CWD-relative loggers — the task this
   file has carried as item 1 for two sessions), **#35** (licence + nightly), **#40**
   (CPU container), **#42** (hardening docs). **#31** (DVC) should be *closed*: the plan
   rejects DVC in favour of a content-hash manifest. **#32** is already contained in
   history and will auto-close.
2. **Phase 3 — the noise floor.** `python -m pipeline.experiment --preset seeds
   --seeds 0,1,2 --yes`. Three of this session's results sat inside ±0.016 and had to be
   reported as "no difference" on the strength of a spread nobody has actually measured.
   This is now the blocking item for every comparison.
3. **The LR *level*, not the schedule.** Round 1 costs a warm-started model 0.066 mAP50
   at `lr0 = 0.01`. The anneal did not fix it because at `local_epochs = 1` there is no
   within-round decay to spread — ultralytics' `LambdaLR` steps once. Try a lower `lr0`
   for a warm-started head, and re-test the anneal at `local_epochs = 4`, which is the
   configuration its argument is actually about.
4. **Phase 1 lever 6** — `evaluate` is 13.8 % of every run, spent on the self-scored
   metric this project already calls the flattering one. `fraction_evaluate < 1.0`.

## A correction worth carrying

`BaseTrainer._get_warmup_iterations` **clamps warmup to `epochs - 1`**. At
`local_epochs = 4` the reference run really did spend three of four epochs in warmup —
but at `local_epochs = 1` there is **no warmup at all**, and every measurement in this
session ran at 1. "Three of every four epochs are warmup" merges two different problems;
the round-1 damage above is the learning-rate *level*.

## Verification

```bash
python -m pytest pipeline/tests -q       # 141
python -m pytest my-project/tests -q     # 36 (40 with the unmerged schedule branch)
python -m pipeline.profile               # where the last run's seconds went
python -m pipeline.verify                # the four pass criteria
python -m pipeline.holdout --evaluate    # the global model on data no vehicle saw
```

## Environment (the part that costs an hour if you forget it)

Venv at `C:\Users\PRANAS\venvs\fl_yolov8`, built on python.org 3.12 — *not* conda; Smart
App Control blocks conda-forge's `_bz2.pyd`. See [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md).
Export `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or every
client trains on CPU at 5.5× the wall clock with no error anywhere.

**Data: done.** All ten shards hold real BDD100K, hardlinked onto the kagglehub cache.
The fleet on disk is currently **1 400 images/vehicle, condition-partitioned, seed 0**.
The attribute index (79 863 images) is cached at `pipeline/.state/attributes.json`.

## The result this project exists to produce, unchanged

6 rounds × 4 local epochs × 6 vehicles × 1 400 images, against a budget-matched
centralised ceiling on the same 201 600 image-visits:

| on 1 000 held-out images | federated | centralised | retained |
|---|---|---|---|
| mAP50 | 0.4173 | 0.4936 | **84.5 %** |
| mAP50-95 | 0.2313 | 0.2770 | 83.5 % |

Nothing this session changed that number — every run here was 1 or 2 local epochs, for
speed of iteration. Re-running it at 6 × 4 with the warm-started head is the first thing
worth doing once the seed spread is known.
