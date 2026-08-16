
# 🚀 Federated Object Detection with YOLOv8 and Flower  

**Collaboratively train YOLOv8 models on distributed datasets while preserving data privacy.**  

---

## 🌐 Table of Contents  
- [🔒 Why Federated Learning?](#-why-federated-learning)  
- [🤖 Why YOLOv8?](#-why-yolov8)  
- [⚙️ Tech Stack](#️-tech-stack)  
- [📥 Installation](#-installation)  
  - [1. Prerequisites](#1-prerequisites)  
  - [2. Create the environment](#2-create-the-environment)  
  - [3. Install dependencies](#3-install-dependencies)  
- [📂 Dataset Preparation](#-dataset-preparation)  
- [🚀 Quick Start](#-quick-start)  
  - [1. Install](#1-install)  
  - [2. Run the federation](#2-run-the-federation)  
  - [3. Monitor Training](#3-monitor-training)  
- [🎛️ Simulation Setup](#️-simulation-setup)  
  - [1. Set Up a Flower Simulation Project](#1-set-up-a-flower-simulation-project)  
  - [4. Clean Up](#4-clean-up)  
- [🏗️ Project Architecture](#️-project-architecture)  
- [🛠️ Troubleshooting](#️-troubleshooting)  
- [📚 References](#-references)  
- [🗺️ Future Roadmap](#️-future-roadmap)  

---

## ▶️ Run it — one command

```powershell
.\scripts\run_pipeline.ps1                 # Windows, demo profile, ~10 min on a 5070 Ti
```
```bash
./scripts/run_pipeline.sh                   # Linux/macOS
```

Tests → shared holdout → dataset → shards → fleet → validation → federated run →
holdout evaluation → pass criteria → comparison. It stops at the first failure
instead of continuing, and prints where the report is.

Compare runs, one command per question:

```bash
python -m pipeline.experiment --preset seeds      --seeds 0,1,2 --yes
python -m pipeline.experiment --preset strategies --strategies fedavg,fedadam --yes
python -m pipeline.experiment --preset partitions --partitions condition,random,dirichlet --yes
python -m pipeline.experiment --preset alpha      --alphas 0.05,0.5,100 --yes
python -m pipeline.compare --last 10              # runs you already have
```

**Full instructions, costs, troubleshooting and how to read the numbers:
[`docs/RUNBOOK.md`](docs/RUNBOOK.md).**

Or without installing anything, on any machine with Docker — the checks only, since
the container has no GPU and no data:

```bash
docker build -t federated-yolov8:cpu .
docker run --rm federated-yolov8:cpu        # 148 passed, 1 skipped
```

**Latest measured result.** 6 vehicles × 1 400 images, condition-partitioned, 6 rounds
× 4 local epochs on an RTX 5070 Ti, scored on 1 000 images no vehicle trained on:

| | federated | centralised, same budget | retained |
|---|---|---|---|
| mAP50 | 0.4173 | 0.4936 | **84.5 %** |
| mAP50-95 | 0.2313 | 0.2770 | 83.5 % |

Both sides made exactly 201 600 image-visits, so the gap measures the method rather
than the budget. Federation costs about 15 % of the achievable accuracy here, in
exchange for never pooling the data. 3 296 s, 82.2 Wh.

---

## 🔒 Why Federated Learning?  
- **Data Privacy**: Sensitive data (e.g., surveillance footage, vehicle sensors) stays on-device.  
- **Bandwidth Efficiency**: Only model gradients (not raw images) are transmitted.  
- **Regulatory Compliance**: Ideal for GDPR, HIPAA, or industry-specific data policies.  
- **Edge Optimization**: Train models directly on edge devices (cameras, drones, IoT sensors).  

---

## 🤖 Why YOLOv8?  
- **State-of-the-Art Performance**: Outperforms YOLOv5 in accuracy and speed.  
- **Multi-Task Support**: Object detection, segmentation, and classification.  
- **Scalability**: Pre-trained models (`yolov8n`, `yolov8s`, etc.) for diverse hardware.  
- **Ease of Use**: Simplified training API and extensive documentation.  

---

## ⚙️ Tech Stack  
- **Frameworks**: [Flower](https://flower.dev) (FL), [Ultralytics YOLOv8](https://ultralytics.com)  
- **Dataset**: [BDD100K](https://bdd-data.berkeley.edu/) (labels committed, images downloaded separately)  
- **GPU Support**: CUDA matched to the card — cu128 for Blackwell (`sm_120`)  
- **Tools**: venv, Git  

---

## 📥 Installation  

### 1. Prerequisites  
- **Python 3.12** — not 3.13: `flwr[simulation]` pulls `ray`, whose Windows
  dependency marker is `python>=3.11,<3.13`.
- **NVIDIA GPU.** Match the CUDA build to the card. Blackwell (RTX 50-series,
  `sm_120`) needs **cu128**; a cu118 wheel has no kernel for it and you get either a
  CUDA error or a silent fall back to CPU.
- **Git**

### 2. Create the environment  
```bash  
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate  
```  

On Windows, build the venv from a **python.org** interpreter — Smart App Control
blocks conda-forge's unsigned stdlib DLLs. See [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md).

### 3. Install dependencies  
```bash  
# PyTorch FIRST, or ultralytics pulls a default-CUDA (CPU-only on Windows) build.  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  

# Confirm the card is actually usable before going further.  
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"  

# Everything else comes from pyproject.toml.  
cd my-project && pip install -e ".[dev]"  
```  

---

## 📂 Dataset Preparation

Full detail — including the download links that are **dead**, so you do not go
looking — is in [`docs/DATASET.md`](docs/DATASET.md).

The repo already ships every shard's labels and split lists
(`my-project/batch/batch_1..10/`, 6 308 train + 1 010 val each). Only the BDD100K
JPEGs are missing. No Kaggle account or token is required:

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('solesensei/solesensei_bdd100k'))"

cd my-project
python scripts/populate_images.py --pool <printed-path>/bdd100k/bdd100k/images/100k
```

`populate_images.py` hardlinks (no extra disk; pool and repo must share a volume, else
`--copy`), is idempotent, and clears the stale Ultralytics `labels/*.cache` files.
Testing? `--batches 1,2 --limit 200` is plenty.

> The Google Drive link this README used to advertise is **gone** — the folder no
> longer resolves. Don't restore it. See the dead-ends table in
> [`docs/DATASET.md`](docs/DATASET.md).

Partitioning is already done: the ten `batch_N/` directories **are** the client
shards, and the server hands each client its own via `CustomBatchStrategy`. Each
shard carries its own `data.yaml` (`nc: 13`); its `path:` is never edited in place —
`task.materialize_data_yaml` writes a gitignored `data.runtime.yaml` beside it.

---

## 🚀 Quick Start  

This project runs through the Flower simulation engine — one command launches the
server and all simulated clients. Full details in [`docs/RUNNING.md`](docs/RUNNING.md).

### 1. Install  
```bash  
cd my-project  
pip install -e .  
```  

### 2. Run the federation  
```bash  
flwr run .  
# override config without editing pyproject.toml:  
flwr run . --run-config "num_server_rounds=5 local_epochs=2"  
```  

### 3. Monitor Training  
Per-module logs land in `my-project/logs/`, one file per process
(`server.<pid>.log`, `client.<pid>.log`, ...) — every simulated client runs in its
own Ray actor and `RotatingFileHandler` is not multi-process safe. They record round
setup, batch-id assignment, weight checksums, and aggregated metrics. Aggregated
per-round metrics also go to `logs/metrics.csv`.  

---

## 🎛️ Simulation Setup  

Flower provides a **simulation engine** to test federated learning on a single machine.

### 1. Set Up a Flower Simulation Project  
```bash  
flwr new my-project --framework PyTorch --username flower  
cd my-project  
pip install -e .  
flwr run .  
```  

### 4. Clean Up  
Use **Ctrl+C** in each terminal to stop the processes.

---

## 🏗️ Project Architecture  

> Run via Flower's app entrypoints (`cd my-project && flwr run .`), **not** standalone
> scripts. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
> [`docs/RUNNING.md`](docs/RUNNING.md) for the full picture.

| Path | Purpose |  
|------|---------|  
| `my-project/pyproject.toml` | Flower app + federation config and hyperparameters (`[tool.flwr.app.config]`). |  
| `my-project/my_project/server_app.py` | `ServerApp` + `CustomBatchStrategy` (FedAvg, batch-id assignment, aggregation). |  
| `my-project/my_project/client_app.py` | `ClientApp` + `FlowerClient` that trains/evaluates YOLOv8 locally. |  
| `my-project/my_project/task.py` | OS detection, model download, runtime `data.yaml` materialization, shard sizing. |  
| `my-project/my_project/get_set_model.py` | Converts weights between the YOLO model and NumPy arrays. |  
| `my-project/utils/logging_setup.py` | Per-module file loggers under `logs/`. |  
| `json_to_yolo/` | BDD100K JSON → YOLO label conversion. |  

---

## 🛠️ Troubleshooting  

| Issue | Solution |  
|-------|----------|  
| **CUDA Out of Memory** | Reduce `BATCH_SIZE` or use `yolov8n`. |  
| **No GPU Detected** | Verify `torch.cuda.is_available()` and reinstall PyTorch with CUDA. |  
| **Dataset Path Errors** | Ensure `data.yaml` paths match the client directory structure. |  
| **Dependency Conflicts** | Use a fresh venv. |  
| **Clients silently train on CPU** | flwr ≥ 1.31 builds its own runtime env and installs the CPU-only torch wheel. Set `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`. |  

---

## 📚 References  
- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com)  
- **Flower**: [Official Documentation](https://flower.dev/docs)  
- **BDD100K**: [Dataset Paper](https://arxiv.org/abs/1805.04687)  

---

## 🗺️ Future Roadmap  
1. **Advanced FL Strategies**: Implement FedProx/FedNova for non-IID data.  
2. **Edge Deployment**: Optimize for NVIDIA Jetson/Raspberry Pi.  
3. **Real-Time Inference**: On-device inference with periodic FL updates.  
4. **Multi-Task Learning**: Add segmentation support with YOLOv8.  

---

Thanks! 😊
