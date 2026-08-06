# CLAUDE.md — federated-yolov8

Project-scoped rules. The workspace file at `~/coding/CLAUDE.md` still applies.

## What this project is

Federated YOLOv8 over BDD100K driving data: a Flower server aggregating per-client
YOLO training, with each client holding its own shard. `pipeline/` reproduces the
whole flow and visualises a simulated vehicle fleet while it runs.

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

Work on a branch. Do not mix an unverified change into a verified one.

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
- One client peaks at **15.9 GB of 16.3 GB** VRAM, so clients train serialised.
  `client-resources.num-gpus = 1.0` is required, not cautious.

## Verification

```bash
python -m pytest my-project/tests -q     # 24 tests
python -m pytest pipeline/tests -q       # 22 tests
python -m pipeline.verify                # the four pass criteria against the last run
```

CI additionally runs an end-to-end federation smoke on CPU and asserts the aggregate
checksum changes between rounds.
