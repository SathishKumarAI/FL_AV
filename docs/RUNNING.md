# Running the Simulation

End-to-end steps to run the federated YOLOv8 training locally with Flower's
simulation engine.

## 1. Environment

```bash
conda create -n fl_yolov8 python=3.10 -y
conda activate fl_yolov8

# PyTorch (CUDA build shown; drop the index-url for CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd my-project
pip install -e .          # installs flwr[simulation], ultralytics, etc. from pyproject.toml
```

## 2. Data layout

Each client shard lives under `my-project/batch/batch_<id>/`:

```
batch/batch_1/
├── data.yaml         # train/val/test + nc + class names
├── images/           # train|val|test images
├── labels/           # YOLO-format labels
├── train.txt val.txt test.txt
```

`data.yaml` ships with a placeholder `path:`; `task.materialize_data_yaml()` writes
a sibling `data.runtime.yaml` (gitignored) carrying the correct local absolute
path, so no manual edit is needed when moving machines and the tracked file is
never mutated.

> Default `batch_id_range` is `(1, 10)` → expects `batch_1` … `batch_10`.

## 3. Configure

Edit `my-project/pyproject.toml`:

```toml
[tool.flwr.app.config]
num_server_rounds = 2
fraction_fit = 0.75
local_epochs = 4
min_clients = 3          # must be <= num-supernodes

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3
options.backend.client-resources.num-gpus = 0   # set 1 only on a CUDA host
```

## 4. Run

```bash
cd my-project
flwr run .
```

Override config without editing the file:

```bash
flwr run . --run-config "num_server_rounds=5 local_epochs=2"
```

## 5. Observe

Per-module logs are written under `my-project/logs/`:

| Log | Content |
|-----|---------|
| `server.log` | round setup, batch-id assignment, weight checksums, aggregated metrics |
| `client.log` | weight load, local train/val, per-client metrics |
| `task.log` | model download, data.yaml path updates, dataset validation |
| `get_set.log` | weight extract/apply + checksums |

Weight checksums on both sides should match each round — that confirms weights
actually transported and applied.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Sim hangs "waiting for clients" | `min_clients` > `num-supernodes`. Lower `min_clients`. |
| `RuntimeError: CUDA ...` / no GPU | Set `num-gpus = 0` in both `client-resources` and `init_args`. |
| `Missing training data.yaml` | `batch_<id>` shard absent or `batch_id_range` mismatched. |
| Parameter count / shape mismatch | Client and server must load the **same** YOLO arch (`yolov8s`). |
| Model download fails | `task.download_model` pulls `yolov8s.pt`; pre-place it in `models/` if offline. |
