# CLAUDE.md — federated-yolov8

Project-scoped rules. The workspace file at `~/coding/CLAUDE.md` still applies.

## What this project is

Federated YOLOv8 over BDD100K driving data: a Flower server aggregating per-client
YOLO training, with each client holding its own shard. `pipeline/` reproduces the
whole flow and visualises a simulated vehicle fleet while it runs.

## Where to look

| Question | File |
|---|---|
| How do I branch, commit, merge? | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Why is mAP low, what do I run next? | [`docs/ML_PLAN.md`](docs/ML_PLAN.md) |
| How do I try another FL algorithm? | [`docs/FL_TECHNIQUES.md`](docs/FL_TECHNIQUES.md) |
| What should I build? | [`docs/BACKLOG_100.md`](docs/BACKLOG_100.md) |
| How do I run any of it? | [`pipeline/README.md`](pipeline/README.md) |
| Where did the last session stop? | [`STATUS.md`](STATUS.md) and `docs/prompts/` |

## Where to change it

Open the one file that owns the thing. Do not read the package to find it.

| Change | File |
|---|---|
| A dashboard panel's look, or any colour, spacing, type | `pipeline/static/app.css` |
| Dashboard markup, a new panel, an element id | `pipeline/static/index.html` |
| Chart axes, ticks, tooltips, sparkline | `pipeline/static/js/chart.js` |
| The fleet grid / vehicle drawer / live polling / run form | `pipeline/static/js/{fleet,drawer,live,control}.js` |
| An HTTP route or what `/api/state` returns | `pipeline/server.py` |
| Which stages exist, what "already done" means, gating | `pipeline/stages.py` |
| How a stage subprocess is run, env, SuperLink handling | `pipeline/runner.py` |
| A path, or an env var a subprocess needs | `pipeline/paths.py` — nowhere else |
| Shard assignment, conditions, partitioning | `pipeline/vehicles.py` |
| What a log line means | `pipeline/logparse.py` |
| The four pass criteria | `pipeline/verify.py` |
| The shared holdout, and scoring the global model on it | `pipeline/holdout.py` |
| The centralised baseline, and the gap to it | `pipeline/baseline.py` |
| What makes a fleet's shards invalid | `pipeline/validate.py` |
| Comparing runs to each other | `pipeline/compare.py` |
| Per-vehicle learning maths (divergence, contribution) | `pipeline/vehicle_metrics.py` |
| The run report | `pipeline/report.py` |
| ⚠ Aggregation strategy — **different branch and prompt** | `my-project/my_project/server_app.py` |
| ⚠ Client training loop, checksums it logs — **ditto** | `my-project/my_project/client_app.py` |
| ⚠ Data yaml, batch path resolution, model loading — **ditto** | `my-project/my_project/task.py` |

Full dashboard map, including the rules that keep it split:
[`pipeline/static/README.md`](pipeline/static/README.md).

Two features span files. Change both halves in the same commit:

- a condition profile → `PROFILES` in `pipeline/vehicles.py` **and** `GLYPHS` in
  `pipeline/static/js/util.js`
- a new artifact kind → the code that writes it **and** `.gitignore` **and**
  `test_generated_paths_are_all_gitignored`

## How work gets done here

**plan → prompt → code → verify.** Not optional, and in that order.

| Step | Artifact | Location |
|---|---|---|
| plan | design, with rejected options and why | `docs/superpowers/specs/` |
| prompt | the brief the code is written from | `docs/prompts/` |
| code | the implementation | wherever it belongs |
| verify | a command whose output proves it | recorded in the commit |

Write the prompt **before** the code, not after. A prompt reconstructed afterwards
records what happened rather than what was intended, and the gap between those two is
the part worth keeping. See `docs/prompts/README.md`.

Work on a branch. Do not mix an unverified change into a verified one. `main` stays
green and is only reached through a squash-merged PR — the full discipline, and the
reasons behind each rule, are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Hard rules

1. **Never commit data.** The BDD100K images are 7.6 GB and live in the kagglehub
   cache; shards hardlink to them. Datasets, checkpoints, MLflow stores, Ray temp
   files and reports are all gitignored, and a test asserts those rules still match
   real paths. If a change can generate a new kind of artifact, add its ignore rule
   in the same commit.
2. **Never commit credentials.** Nothing here needs any: kagglehub downloads
   anonymously. No `kaggle.json`, no `.env`, no tokens — not read, not copied, not
   echoed into a log. Hosted trackers (W&B, Comet, Neptune) are rejected for this
   reason, not on quality.
3. **`pipeline/` must not modify `my-project/`.** It invokes its scripts as
   subprocesses and reads its outputs. Enforced by
   `pipeline/tests/test_pipeline.py::test_pipeline_never_writes_into_my_project`.
4. **Assemble before building.** MLflow owns metrics and history; the Ray Dashboard
   owns actor and GPU internals; Ultralytics already ships an MLflow callback. Write
   only what none of them can do. A bespoke dashboard was proposed once and correctly
   rejected as reinventing the wheel.
5. **Fail loudly.** A failed stage halts the chain. This project's history is a
   catalogue of silent no-ops — see below — so nothing may continue past a failure or
   let a no-op look like success.

## Silent failures this project has already shipped

Read these before assuming a green run means a working one.

| | What looked fine | What was happening |
|---|---|---|
| B4 | rounds completed, metrics logged | clients returned the weights they were *sent*; FedAvg averaged its own input |
| B9 | server logged two shard assignments | `FedAvg` shares one `FitIns`, so every client got the **last** `batch_id` and the fleet trained one shard |
| B7 | federation ran to completion | checkpointing silently skipped every round |
| — | run succeeded, checksums identical | shard too small for the batch size: **no optimizer step happened at all** |
| — | `metrics.csv` showed 6 308 examples | only 10 images existed; `num_examples` is FedAvg's weight |

**The single most useful signal is the round-over-round aggregate checksum.** Equal
consecutive values mean nothing is being learned, whatever the metrics say.

## Environment traps

- Use the venv on **python.org 3.12**, not conda: Smart App Control blocks
  conda-forge's `_bz2.pyd`. See `docs/ENV_WINDOWS.md`.
- Export `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or flwr
  builds its own runtime env with the CPU-only torch wheel and every client trains on
  CPU at ~5.5x wall clock, silently.
- `flwr run` **rewrites `my-project/pyproject.toml`**, commenting out
  `[tool.flwr.federations]`. Restore it; never commit the rewritten form. The pipeline
  does this automatically after every run.
- Blackwell (`sm_120`) needs torch **cu128**. cu118 has no kernel for it.
- VRAM depends on the profile: a full 6 308-image shard peaks at **15.9 GB of 16.3 GB**,
  the 300-image demo at ~5 GB. `client-resources.num-gpus = 1.0` serialises clients,
  which is required at full scale and leaves real headroom at demo scale.
- The detached SuperLink caches the CWD **and environment** of whichever `flwr run`
  started it. The pipeline kills it before every federation for that reason.
- Condition partitioning is only real while the condition has images: `overcast
  residential` has 1 419 in all of BDD100K. Asking for more per vehicle silently tops up
  with random images and turns a non-IID run into a nearly-IID one.

## Agile, in the way that actually matters here

Small vertical slices that each end in something demonstrable — a passing test, a moved
metric, a screenshot. A slice that cannot be demonstrated is not done, it is in progress.

Every increment leaves three artifacts behind: the **prompt** it was built from, the
**verification** that it works, and an updated **STATUS.md** so the next session starts
from fact rather than archaeology. Experiments live on `exp/` branches and are allowed to
be thrown away — but their *result* gets written down somewhere permanent before the
branch dies.

## Verification

```bash
python -m pytest my-project/tests -q     # 31 tests
python -m pytest pipeline/tests -q       # 59 tests
python -m pipeline.verify                # the four pass criteria against the last run
python -m pipeline.holdout --evaluate    # the global model on data no vehicle saw
```

CI additionally runs an end-to-end federation smoke on CPU and asserts the aggregate
checksum changes between rounds.
