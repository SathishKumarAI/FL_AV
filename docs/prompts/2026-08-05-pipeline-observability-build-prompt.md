# Build prompt — pipeline + observability

**Design:** [`2026-08-05-pipeline-observability-design.md`](../superpowers/specs/2026-08-05-pipeline-observability-design.md)
**Branch:** `feat/pipeline-observability`

## Goal

One command reproduces the whole federated-YOLOv8 flow on a fresh machine — environment
check, dataset, shard population, single-client GPU sanity, federated run, verification —
and while it runs you can watch what is happening: which round, which shard, whether the
global weights are actually moving. Observability comes from tools that already exist
(MLflow, Ray Dashboard), not from anything written here.

## Hard constraints

1. **Do not modify `my-project/`.** The pipeline invokes it as subprocesses and reads its
   output files. It must not import its internals, edit its source, or write into its
   tree beyond what those subprocesses do themselves. A test enforces this.
2. **Do not build a dashboard.** No HTTP server, no HTML, no charting code, no SSE. If a
   view is wanted, it comes from MLflow or the Ray Dashboard. This constraint exists
   because the first design did exactly that and was rejected as reinventing the wheel.
3. **No credentials, ever.** The pipeline reads no `kaggle.json`, no `.env`, no token of
   any kind, and echoes none. kagglehub downloads anonymously — verified.
4. **No data to GitHub.** Dataset images, checkpoints, MLflow stores, Ray temp files and
   run artifacts must all be unable to be committed. Add the ignore rules in the same
   change that can produce the files.
5. **Fail loudly.** A stage that fails halts the chain. Never continue past a failure and
   never let a no-op look like success — most of this project's history is silent no-ops
   (B4, B9, the zero-optimizer-step round).
6. **No new heavyweight dependency.** MLflow and `ray[default]` are approved and
   installed. Prefect and the hosted trackers are rejected; do not reintroduce them.

## Inputs — already verified, do not re-derive

- Versions: `torch 2.11.0+cu128`, `flwr 1.33.0`, `ultralytics 8.4.115`, `ray 2.55.1`,
  `mlflow 3.15.1`. flwr and ultralytics are the latest PyPI releases as of 2026-08-05.
- Data: all 10 shards populated, 6 308 train + 1 010 val each, hardlinked from the
  kagglehub cache at
  `~/.cache/kagglehub/datasets/solesensei/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/100k`
  (note the doubled `bdd100k`).
- One client peaks at **15.9 GB of 16.3 GB** VRAM, so clients must stay serialised
  (`client-resources.num-gpus = 1.0`). 0.5 would OOM.
- `flwr run` needs `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`, or every client
  silently trains on CPU at 5.5x the wall clock.
- `flwr run` rewrites `pyproject.toml`, commenting out `[tool.flwr.federations]`. It must
  be restored, never committed in the rewritten form.
- Ray Dashboard: starting a head node first and exporting `RAY_ADDRESS` makes flwr's
  `ray.init(runtime_env=..., include_dashboard=False)` **attach** to it. Verified:
  `GPU: 1.0`, `nodes: 1`, dashboard `/api/version` → `200`.
- Log markers to parse (they already exist; do not add logging to `my-project`):
  `Aggregated parameters with checksum`, `Received weights with checksum`,
  `Sending back weights with checksum`, `Starting local training with batch_id`,
  plus `logs/metrics.csv` rows and `checkpoints/global_*.pt`.

## Deliverables

```
pipeline/
├── stages.py         # six stages: detection, skip logic, command construction
├── runner.py         # CLI: sequencing, confirm-gating, exit codes
├── mlflow_sink.py    # parse my-project logs/metrics -> MLflow runs
├── observability.py  # start/stop MLflow server + Ray head, print URLs
├── requirements.txt  # mlflow, ray[default] -- kept OUT of my-project's deps
├── README.md         # how to run it
├── tests/
└── docs/ARCHITECTURE.md   # + mermaid UML: component, sequence, stage state machine
```

Stages, with skip conditions:

| Stage | Runs | Skips when | Confirm-gated |
|---|---|---|---|
| `env` | torch/CUDA probe in-process | never | no |
| `dataset` | `kagglehub.dataset_download` | pool has 70 000 train + 10 000 val | yes |
| `populate` | `scripts/populate_images.py --pool …` | every shard's image count matches its split list | no |
| `sanity` | `yolo detect train … epochs=1` | marker from a previous pass | yes |
| `federate` | `flwr run . --stream --run-config …` | never | yes |
| `verify` | asserts the four pass criteria | never | no |

MLflow structure: one parent run per invocation, a nested run per stage, per-round
metrics on `federate`.

## Definition of done

```bash
python -m pytest pipeline/tests -q          # all pass
python -m pipeline.runner --list            # shows six stages with live skip state
python -m pipeline.runner --stages env,verify   # runs ungated stages, exits 0
python -m pytest my-project/tests -q        # still 24 passed -- nothing regressed
git status --short                          # no dataset, checkpoint or store files
```

Plus one real end-to-end federated run driven by the pipeline, with its MLflow run
showing per-round mAP50 and aggregate checksums that differ between rounds.

## Out of scope

Scheduling, retries, multi-machine deployment, run comparison across machines, and the
SuperLink/TLS deployment path. Any bespoke UI.
