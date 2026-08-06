# Pipeline + observability — design

**Date:** 2026-08-05 · **Branch:** `feat/pipeline-observability` · **Status:** approved

## Problem

Reproducing this project end to end means running six things in order, each with a
trap that fails silently rather than loudly: the wrong torch build falls back to CPU,
the dataset pool nests its train split, flwr rebuilds an isolated env with a CPU-only
wheel, and a round can complete without a single optimizer step. All of it was
established by hand this session and lives only in prose in `docs/GPU_TESTPLAN.md`.

Two goals: **one command reproduces the flow**, and **you can watch it happen** —
which round is running, on which shard, whether the global weights are actually
moving.

## Decision: assemble, don't build

The first draft of this design was a bespoke HTTP server, an SSE event bus and a
hand-drawn SVG dashboard. That was rejected as reinventing the wheel, correctly.

What already exists, verified in the installed packages rather than assumed:

| Capability | Provided by | Evidence |
|---|---|---|
| Per-client training curves | Ultralytics' own MLflow callback | `ultralytics/utils/callbacks/mlflow.py` ships in 8.4.115, alongside tensorboard/clearml/comet/dvc/neptune/wandb |
| Experiment tracking UI, run comparison, artifact store | MLflow 3.15 | local server, no account |
| Live actor / GPU / timeline view of the federation | Ray Dashboard | `ray[default]`, dashboard answered `200` on `/api/version` |
| Federation orchestration | flwr itself | already in use |

**Flower has no dashboard of its own** — its guidance is to use TensorBoard. The
"Flower observability" recollection is Ray's, surfacing through flwr's simulation
backend.

### Ray Dashboard without touching the repo

`RayBackend.init_ray()` hardcodes `include_dashboard=False` and the TOML schema only
accepts `num-cpus`/`num-gpus`/`logging-level`/`log-to-driver`, so the dashboard cannot
be switched on through config. But the call is guarded by `if not ray.is_initialized()`.

Verified: starting a head node first and exporting `RAY_ADDRESS` makes flwr's exact
call — `ray.init(runtime_env=..., include_dashboard=False)` — **attach** to the running
cluster (`GPU: 1.0`, `nodes: 1`) instead of creating a dashboard-less one. No source
change, in this repo or in flwr.

### Rejected

- **Prefect** — 60 packages including redis, docker and asyncpg into the same venv as
  cu128 torch, to sequence six linear steps. MLflow's parent/nested runs already give
  status, duration and metrics per stage. Revisit only if scheduling or retries are
  genuinely needed.
- **W&B, Comet, Neptune, ClearML-hosted** — all ship run data off-machine and require
  API keys. Ruled out by the project's own constraints, independent of quality.
- **Bespoke dashboard** — the original proposal. Strictly more code for strictly less
  function than MLflow's UI.

**What this costs:** the FedAvg weight-flow diagram (server → per-client → aggregate)
becomes an MLflow metric series rather than a bespoke visual. Less vivid, far less
code. Accepted trade.

## Architecture

New top-level `pipeline/`. It **invokes** `my-project` and **reads** its outputs. It
never imports its internals and never modifies it — that isolation is the point, and
is enforced by a test.

```
pipeline/
├── stages.py        # the six stages: detect, skip, run, gate
├── runner.py        # CLI: sequencing, confirm-gating, exit codes
├── mlflow_sink.py   # parse my-project's logs/metrics -> MLflow
├── observability.py # start/stop MLflow server + Ray head, print URLs
├── tests/
└── docs/            # ARCHITECTURE.md + mermaid UML
```

### Stages

| Stage | Runs | Skips when | Gated |
|---|---|---|---|
| `env` | in-process torch/CUDA probe | never (cheap, informative) | no |
| `dataset` | `kagglehub.dataset_download` | pool holds 70 000 train + 10 000 val | **yes** |
| `populate` | `scripts/populate_images.py --pool …` | every shard's image count matches its split list | no |
| `sanity` | `yolo detect train … epochs=1` | marker file from a previous pass | **yes** |
| `federate` | `flwr run . --stream --run-config …` | never — it is the point | **yes** |
| `verify` | asserts the four pass criteria | never | no |

Gated stages cost real time or GPU and require `--yes` or an interactive confirm.
Every stage is a subprocess with `cwd=my-project`, which also keeps flwr's sticky
SuperLink CWD correct. A non-zero exit fails the stage and **halts the chain** — no
silent continue, given how much of this project's history is silent no-ops.

### Data flow

```
runner.py ──subprocess──> my-project (yolo / flwr / populate)
    │                          │
    │                          ├─> logs/*.log, logs/metrics.csv, checkpoints/
    │                          └─> Ultralytics MLflow callback ──> MLflow
    │
    └── mlflow_sink.py ──reads those files──> MLflow (FL-level runs)
```

Two independent writers to MLflow: Ultralytics logs *client training* by itself; the
sink logs *federation-level* facts (round, aggregate checksum, aggregated mAP,
per-client shard) parsed from the log markers that already exist —
`Aggregated parameters with checksum`, `Starting local training with batch_id`,
`Received`/`Sending back weights with checksum`, plus `metrics.csv` rows.

Structure in MLflow: one parent run per pipeline invocation, nested runs per stage,
and per-round metrics on the `federate` run so the UI charts them over rounds.

### Observability surfaces

- **MLflow UI** (`:5000`) — metrics, params, run comparison, artifacts. The primary view.
- **Ray Dashboard** (`:8265`) — actors, GPU utilisation, per-actor logs, during the
  federation only.

Both bind loopback. Neither sends anything off the machine.

## Safety

- `.gitignore` additions so the dataset, checkpoints, MLflow store, Ray temp files and
  any credential file cannot be committed. A test asserts the rules match real paths.
- The pipeline handles **no credentials**. kagglehub downloads anonymously; nothing
  reads, copies or echoes `kaggle.json` or `.env`.
- MLflow's backing store lives under `pipeline/` and is gitignored — it holds metrics
  and model artifacts, neither of which belongs in the repo.

## Testing

Unit tests over the logic that can be wrong quietly:

- skip-detection per stage (populated vs partially populated vs empty)
- log-line parsing → the exact checksum/shard values from a captured log fixture
- stage chain halts on failure and does not run later stages
- **isolation guard**: no file under `pipeline/` writes to `my-project/`

The GPU stages are not unit-tested; `verify` is what asserts them, and it is the same
four criteria the CI smoke job already uses.

## Out of scope

Scheduling, retries, multi-machine deployment, and comparing runs across machines.
`docs/DEPLOYMENT.md`'s SuperLink/TLS path stays a separate exercise.
