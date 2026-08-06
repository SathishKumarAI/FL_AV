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

`env` → `dataset` → `populate` → `holdout` → `fleet` → `sanity` → `federate` →
`evaluate` → `verify` → `baseline`

Each detects whether its work is already done and skips if so, so re-runs are cheap.
`dataset`, `sanity`, `federate` and `baseline` are **gated**: they cost real time or
GPU and need `--yes` (or the confirm box) before they start. A failed stage halts the
chain.

`holdout` runs **before** `fleet`, and the order is load-bearing: it carves a val
slice that `build_fleet` then subtracts from the pool, so no vehicle can train or
self-evaluate on it. A holdout carved afterwards would already be inside somebody's
val split, and the "global" metric measured on it would be partly self-referential.

## The honest metric, and the ceiling

Every per-client number is a vehicle scoring itself on its own distribution, so those
numbers are not comparable with each other and their average is not a global metric.
Two stages fix that:

```bash
python -m pipeline.holdout --build --size 1000    # carve it (before the fleet)
python -m pipeline.holdout --evaluate             # score every global checkpoint on it
python -m pipeline.baseline --rounds 6 --local-epochs 4   # the centralised ceiling
```

`baseline` trains one model on the pooled union of every vehicle's images for
`rounds × local_epochs` epochs — exactly the image-visits the fleet makes — and scores
it on the same holdout, so the gap between federated and centralised is a fair
comparison rather than a flattering one.

## Partitioning and strategies

```bash
--partition condition|random|mixed|dirichlet   --alpha 0.3
--strategy  fedavg|fedprox|fedadam|fedyogi|fedadagrad|fedavgm|fedmedian|...
```

Both are registries: `@partitioner("name")` in `pipeline/vehicles.py`, and
`server_app.STRATEGIES` probed from what this Flower build exports. Adding a
partitioner is one function; the CLI choices, the dashboard menu and the validation
all follow from the registration. An unknown name is rejected before any subprocess
starts, never quietly replaced with the default.

## Checking the data, comparing the runs

```bash
python -m pipeline.validate      # six ways a fleet can be quietly wrong
python -m pipeline.compare       # the last 5 runs, holdout number first
```

`validate` is read-only and runs as a stage between `fleet` and `sanity`: it checks
label coverage, listing integrity, cross-shard leakage, train/val leakage, holdout
containment and label sanity. It reports and refuses to repair — a validator that
edits your data hides the bug that produced it.

`compare` warns when the runs it is showing differ in more than one setting, because
then the difference in their numbers cannot be attributed to any of them.

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
