# pipeline — run the flow, watch the fleet

Reproduces the whole federated-YOLOv8 flow and explains it while it runs. Simulates a
fleet of vehicles, each learning from a different slice of the world.

**It never modifies `my-project`.** It runs its scripts as subprocesses and reads its
outputs — enforced by a test, not just promised.

## Install

Everything but these extras is already in the project venv:

```bash
pip install -r pipeline/requirements.txt
```

## Run

```bash
python -m pipeline.runner --list                     # what would run, what would skip
python -m pipeline.runner --stages env,fleet         # a subset
python -m pipeline.runner --all --profile demo --vehicles 6 --yes
```

Or drive it from the browser:

```bash
python -m pipeline.server        # http://127.0.0.1:8800
```

Two views: **Control** to configure and launch, **Live** to watch the fleet, the
weight flow and the GPU.

## The other two UIs

This component deliberately does not reimplement them.

```bash
mlflow ui --backend-store-uri pipeline/mlruns --port 5000    # metrics, history, comparison
ray start --head --dashboard-host 127.0.0.1                  # actors, GPU internals
python -m pipeline.runner --all --ray-address 127.0.0.1:6379 # ...then attach to it
```

The Ray trick matters: flwr hardcodes `include_dashboard=False`, but only calls
`ray.init()` when Ray is not already running. Start a head node first and the
federation attaches to it, dashboard included, with no source change anywhere.

## Profiles

| Profile | Images/vehicle | imgsz | Wall clock |
|---|---|---|---|
| `demo` | 300 | 320 | minutes — watchable |
| `full` | 6 308 | 640 | hours |

Vehicles train **serialised** — one client peaks at 15.9 GB of 16.3 GB VRAM, so
concurrency would OOM. Time scales with vehicle count.

## Stages

`env` → `dataset` → `populate` → `fleet` → `sanity` → `federate` → `verify`

Each detects whether its work is already done and skips if so, so re-runs are cheap.
`dataset`, `sanity` and `federate` are **gated**: they cost real time or GPU and need
`--yes` (or the confirm box) before they start. A failed stage halts the chain.

## Output

Every run writes `pipeline/reports/<timestamp>/`:

- `report.html` — self-contained, opens anywhere, no CDN
- `report.md` — the same content, diffable
- `report.json` — the raw data both were rendered from

Covering inputs (config, vehicle→condition map, shard sizes), what was learned
(per-round aggregate checksums, mAP), what it cost (GPU energy in Wh, peak VRAM,
stage timings), and outputs (checkpoints).

## Tests

```bash
python -m pytest pipeline/tests -q
```

Covers the parts that fail quietly: checksum parsing including negative values,
detection of a federation that did **not** learn, disjoint vehicle slices, energy
integration, that expensive stages are gated, and that nothing here writes into
`my-project` or commits generated data.

## Architecture

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, sequence and stage
state machine as UML.
