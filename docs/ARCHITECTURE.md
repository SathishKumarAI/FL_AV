# Architecture

Federated object detection: **YOLOv8** (Ultralytics) trained across simulated
clients with the **Flower** framework. Each client owns a different data shard
(`batch_N`); only model weights — never images — cross the wire.

## Runtime topology

```
                    flwr run  (Flower Simulation Engine)
                              │
                ┌─────────────┴─────────────┐
                │     ServerApp (server_fn)  │
                │  CustomBatchStrategy(FedAvg)│
                │  - assigns unique batch_id  │
                │  - aggregates weights       │
                │  - tracks per-client OS     │
                └─────────────┬─────────────┘
          round: send global weights + {batch_id, local_epochs}
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
          SuperNode 0    SuperNode 1    SuperNode 2     (num-supernodes)
          ClientApp      ClientApp      ClientApp
          (client_fn)    (client_fn)    (client_fn)
          YOLO.train on  YOLO.train on  YOLO.train on
          batch_<id>     batch_<id>     batch_<id>
                │             │             │
          return updated weights + metrics (mAP, precision, recall)
                └─────────────┼─────────────┘
                              ▼
                   FedAvg weighted aggregation
```

## Federated round

1. **Server → clients**: global weights + per-client config `{batch_id, local_epochs}`.
   `CustomBatchStrategy.configure_fit` hands each client a *unique* `batch_id` so no
   two clients train on the same shard in a round.
2. **Client fit**: `set_weights` loads the global model, `YOLO.train` runs
   `local_epochs` on `batch/batch_<id>/data.yaml`, `get_weights` extracts the result.
3. **Clients → server**: updated weights + metrics (`precision`, `recall`, `mAP50`,
   `mAP50-95`, `fitness`, `os`).
4. **Aggregate**: `aggregate_fit` runs FedAvg (weighted by `num_examples`) and logs a
   weight checksum so weight transport can be verified end-to-end.
5. **Evaluate**: `configure_evaluate` → `YOLO.val` per client → `aggregate_evaluate`.

Repeat for `num_server_rounds`.

## Actual file map

> The code runs through Flower's app entrypoints (`flwr run`), **not** standalone
> `server.py` / `client.py` scripts.

| Path | Role |
|------|------|
| `my-project/pyproject.toml` | Flower app + federation config; `[tool.flwr.app.config]` hyperparameters |
| `my-project/my_project/server_app.py` | `ServerApp`, `server_fn`, `CustomBatchStrategy` (FedAvg + batch assignment + OS tracking) |
| `my-project/my_project/client_app.py` | `ClientApp`, `client_fn`, `FlowerClient` (YOLO train/eval) |
| `my-project/my_project/task.py` | OS detection, model download, data.yaml path handling, dataset validation, GPU batch-size heuristic |
| `my-project/my_project/get_set_model.py` | `get_weights` / `set_weights` between YOLO `nn.Module` and NumPy arrays |
| `my-project/utils/logging_setup.py` | Per-module file loggers under `logs/` |
| `my-project/models/` | `yolov8s.pt` initial weights, `yolo8n.yaml` config |
| `my-project/batch/batch_<id>/` | Per-client data shard (`data.yaml`, `images/`, `labels/`, split `.txt` files) |
| `json_to_yolo/` | BDD100K JSON → YOLO label conversion notebooks/scripts |

## Configuration

Hyperparameters live in `my-project/pyproject.toml` under `[tool.flwr.app.config]`
and are read by `server_fn` via `context.run_config`:

| Key | Meaning |
|-----|---------|
| `num_server_rounds` | Number of FL rounds |
| `fraction_fit` | Fraction of available clients sampled per round |
| `local_epochs` | YOLO epochs each client runs per round |
| `min_clients` | Minimum clients to start a round — **must be ≤ `num-supernodes`** |

Federation sizing is under `[tool.flwr.federations.local-simulation]`
(`num-supernodes`, per-client `num-cpus` / `num-gpus`). Override at runtime:

```bash
flwr run . --run-config "num_server_rounds=5 local_epochs=2"
```

## Class taxonomy (nc = 13)

`person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light,
traffic sign, trailer, other person, other vehicle` — the BDD100K detection set.

## Known constraints

- Client `num_examples` comes from `task.count_shard_examples` (the split list), so
  FedAvg weighting is proportional to real shard size.
- `num-gpus` is `1.0` per client, which serialises clients on a single card. Drop it
  to `0.5`/`0.33` only once you know one client's real VRAM footprint, or to `0` for
  a CPU-only host.
- `data.yaml` `path:` is not mutated. `task.materialize_data_yaml` writes a
  gitignored `data.runtime.yaml` sibling with the local absolute path, and training
  and validation point at that.
