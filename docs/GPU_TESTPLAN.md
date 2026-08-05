# GPU Test Plan — first real hardware run

Target host: **RTX 5070 Ti, 16 GB, driver 610.47, Windows 11**.

> **STATUS 2026-08-05 — executed. All phases through Phase 4 pass on the GPU.**
> B1–B6 are fixed, plus three defects this run exposed that the plan below did not
> predict (B7–B9). Results in §6. The environment is a **venv on python.org 3.12**,
> not conda — see [ENV_WINDOWS.md](ENV_WINDOWS.md) for why.

Related: [ENV_WINDOWS.md](ENV_WINDOWS.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
[RUNNING.md](RUNNING.md) · [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)

---

## 0. Blockers

Ordered by what stops the run first. Details and fixes in §5.

| # | Blocker | Effect | Status |
|---|---------|--------|--------|
| B1 | `batch/*/images/` is empty (labels + split lists committed, JPEGs not) | Every client dies: "no images found" | fixed — fixture, §2 |
| B2 | Docs install torch `cu118` | 5070 Ti is Blackwell `sm_120`; cu118 wheels have no kernel for it — CUDA error or silent CPU fallback | fixed — cu128 |
| B3 | `pyproject.toml` sets `num-gpus = 0` in both `client-resources` and `init_args` | Ray hides the GPU; the whole "GPU test" runs on CPU and looks fine | fixed — and the key was `init-args`, not `init_args`; the old spelling was ignored outright |
| B4 | Client extracts post-training weights from the **wrong module object** | Nothing the client learns ever reaches the server | **confirmed and fixed** |
| B5 | Server model is `nc=80` (COCO `yolov8s.pt`), trainer rebuilds `nc=13` from `data.yaml` | Head shapes disagree between server and client | **confirmed and fixed** |
| B6 | Stale `labels/*.cache` from the previous machine | Ultralytics trusts the cache and resolves paths that don't exist here | fixed at the root — the caches were **tracked in git**; removed and ignored |

### Found by running it — not predicted by this plan

| # | Blocker | Effect |
|---|---------|--------|
| B7 | `_save_global_model` (`server_app.py`) built a third `nc=80` model | `set_weights` returns False, checkpoint silently skipped, pass criterion 4 unreachable |
| B8 | Every Ray actor wrote the same `logs/client.log` | `RotatingFileHandler` is not multi-process safe; records interleave and vanish, so the log lied about which shard a client trained |
| B9 | **`configure_fit` mutated a shared config dict** | `FedAvg` hands ONE `FitIns` to every client. Writing `config["batch_id"]` per client overwrote it for all — last write won. The server logged two distinct assignments while both clients trained the **same shard**. Federated partitioning was a no-op. |

B9 is the worst of the three: it is invisible from the server's own logs, which is
exactly why `tests/test_batch_assignment.py` now guards it.

---

## Phase 1 — environment + the two-minute correctness probe

Cost: ~15 min, mostly download. Proves CUDA works and answers B4/B5.

```powershell
conda create -n fl_yolov8 python=3.11 -y
conda activate fl_yolov8

# Blackwell (sm_120) needs a cu128 build. NOT cu118.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 2.7+ / 12.8 / (12, 0)   <- (12,0) confirms sm_120 is supported

cd my-project
pip install -e .
```

Then the probe. This is the whole experiment for B4 + B5:

```python
# probe.py  -- run from my-project/, needs one populated shard (see Phase 2 fast path)
from ultralytics import YOLO
from my_project.get_set_model import get_weights

y = YOLO("models/yolov8s.pt")
before_obj, before_n = y.model, len(get_weights(y.model))
y.train(data="batch/batch_1/data.runtime.yaml", epochs=1, imgsz=320, batch=4, device=0)
print("same module object after train:", y.model is before_obj)   # expect False -> B4 real
print("state tensors before/after:", before_n, len(get_weights(y.model)))
print("head nc before/after:", 80, y.model.nc)                    # expect 80 -> 13 -> B5 real
```

- `same module object: False` ⇒ **B4 confirmed.** `client_app.py` keeps
  `self.model` from `__init__` and reads it back with `get_weights(self.model)`
  after `self.yolo.train(...)`. Ultralytics reassigns `yolo.model` from the
  best/last checkpoint at the end of `train()`, so `self.model` is an orphan: the
  client returns the weights it was *sent*, never the weights it *trained*. Round
  2 onward it also trains from the orphan's stale state. FedAvg then averages
  identical inputs — the federation runs, logs metrics, and learns nothing.
- `head nc 80 -> 13` ⇒ **B5 confirmed.** Fixing B4 alone makes it visible as a
  `set_weights` count/shape mismatch, because the server still holds an `nc=80`
  model. Both must land together.

---

## Phase 2 — data

The repo has all 10 shards' labels (7 318 files each: 6 308 train / 1 010 val)
and the split lists. Only the BDD100K JPEGs are missing.

### Where the images are *not* (searched 2026-08-04)

Before re-downloading 6.3 GB, this is what was already checked and ruled out.

| Location | Result |
|----------|--------|
| Google Drive, `sathishkumar786.ml@gmail.com` | **Nothing.** Full enumeration of every folder and every archive mime-type: zero zip/tar/7z files in the account, no folder named bdd/yolo/dataset/FL_AV. Only doc-type files. |
| The README's "Download Dataset" Drive link (folder `1R-lelZR3LBgeHfMlRR_OhOIzfUuxPBcZ`) | **Dead** — "requested entity was not found". Either deleted or owned by an account that no longer shares it. The README still advertises it; see the gap table. |
| `origin/images` branch | 0 images despite the name. |
| `origin/laptop_copy` branch | 450 JPEGs — exactly 10 per split per shard. A toy fixture (matches the `batch34/` dirs, 20 labels each), **not** the dataset. Useful as a CI fixture, useless for training. |

So the images have to come from an external source. If they exist anywhere it
would be another Drive account, an external drive, or the old
`C:\Users\sathish\Downloads\FL_ModelForAV\` machine that the committed
`data.yaml` and `full_data_run/detect/train2/args.yaml` paths point at.

**Fast path (do this first — ~1 GB, unblocks Phases 1 and 3):** grab only the
BDD100K **val** set (10 k images) and build a 2-shard smoke federation from it.
Full-scale can download in the background while you debug.

```powershell
python scripts/populate_images.py --pool D:\bdd100k\images\100k --batches 1,2 --limit 200
```

**Full path:** BDD100K "100K Images" (~5.3 GB train, ~1 GB val). Sources, in
order of preference:

1. Official — <https://bdd-data.berkeley.edu/> (free account, then *100K Images*).
2. The ETH mirror the BDD100K docs list (`dl.cv.ethz.ch/bdd100k/data/`) —
   **did not resolve when checked on 2026-08-04**; confirm before relying on it.
3. Kaggle: `kaggle datasets download -d solesensei/solesensei_bdd100k`.

You only need `bdd100k/images/100k/{train,val}/`. The `labels/` in those archives
are the raw JSON — ignore them, this repo's YOLO `.txt` labels are already
converted and committed.

```powershell
python scripts/populate_images.py --pool D:\bdd100k\images\100k   # all 10 shards
```

The script hardlinks (near-zero disk, pool must be on the same volume as the
repo), is idempotent, and deletes the stale `labels/*.cache` (B6). `--copy`
forces a real copy across volumes. Verify with `--self-check`.

---

## Phase 3 — single-client GPU sanity, no federation

Before paying for a 3-client simulation, prove one YOLO run trains on this card.

```powershell
cd my-project
yolo detect train data=batch/batch_1/data.runtime.yaml model=models/yolov8s.pt `
  epochs=1 imgsz=640 batch=8 device=0 workers=2 amp=True
```

Watch for: `AMP: checks passed`, GPU utilisation non-zero in `nvidia-smi`,
box/cls/dfl losses all finite and falling.

| Symptom | Cause |
|---------|-------|
| `no kernel image is available for execution` | B2 — wrong torch build, reinstall cu128 |
| AMP check hangs | it downloads `yolo11n.pt`; pre-place it or run with `amp=False` |
| Dataloader hangs/crashes on Windows | `workers=2` or `workers=0` |
| `Dataset ... images not found` | Phase 2 incomplete, or a stale `.cache` survived |

---

## Phase 4 — federated simulation on the GPU

Config for a **16 GB single-card** host. `num-gpus = 1.0` per client makes Ray
schedule clients **one at a time** on the GPU — slower but no OOM. Only drop to
`0.5`/`0.33` once you know one client's real footprint from Phase 3.

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 2
options.backend.client-resources.num-cpus = 4
options.backend.client-resources.num-gpus = 1.0
options.backend.init_args.num_cpus = 8
options.backend.init_args.num_gpus = 1
```

```powershell
flwr run . --run-config "num_server_rounds=2 local_epochs=1 min_clients=2"
```

**Pass criteria — all four, not just "it finished":**

1. `logs/server.log` weight checksum **changes between rounds**. If round 1 and
   round 2 checksums are equal, B4 is still live and the federation is a no-op.
2. Client-sent checksum ≠ client-received checksum in `logs/client.log`.
3. `logs/metrics.csv` has fit and evaluate rows with non-zero mAP50.
4. `checkpoints/global_round_2.pt` and `global_last.pt` exist and reload:
   `YOLO("checkpoints/global_last.pt").val(data=...)`.

Then scale: `num-supernodes=3`, `num_server_rounds=5`, `local_epochs=2`,
full shards. Expect hours, not minutes — 6 308 images × 3 clients × epochs on one
card is serialised by the `num-gpus = 1.0` setting.

---

## Phase 5 — FedProx and the distributed engine

Only after Phase 4 passes on FedAvg.

```powershell
flwr run . --run-config "strategy='fedprox' proximal_mu=0.1 num_server_rounds=3"
```

Compare `logs/metrics.csv` against the FedAvg run. The deployment path
(SuperLink + SuperNodes + TLS, [DEPLOYMENT.md](DEPLOYMENT.md)) has never been run
either; it needs Docker Desktop + the NVIDIA Container Toolkit and is a separate
exercise — do not mix it into the first GPU bring-up.

---

## 5. Gaps and fixes

### Must fix before Phase 4 means anything

| ID | Where | Fix |
|----|-------|-----|
| B4 | `my_project/client_app.py` — `fit()` step 6 | Read back from the live model: `self.model = self.yolo.model` immediately after `self.yolo.train(...)`, then `get_weights(self.model)`. Same in `evaluate()` after `val()`. |
| B5 | `client_app.py` `__init__` + `server_app.py` `server_fn` | `self.model.nc = 13` is a no-op — it renames an attribute, it does not rebuild the 80-class head. Both sides must construct the *same* `nc=13` arch: `YOLO("models/yolov8s-13.yaml").load("models/yolov8s.pt")`, where the yaml is `models/yolo8n.yaml` (already `nc: 13`) copied to an `-s`-scaled filename. `set_weights`' `strict=True` will confirm it. |
| B3 | `pyproject.toml` | See Phase 4 block. |
| B6 | shards | Handled by `scripts/populate_images.py`. |

### Worth fixing while you are in there

| Where | Gap |
|-------|-----|
| `client_app.py` `fit()` | `yolo.train()` is called with no `batch=`, so Ultralytics uses 16 regardless of the card. `task.get_optimal_batch_size()` exists for exactly this and **is never called** — wire it in, or pass `batch` from `run_config`. |
| `client_app.py` `fit()` | No `workers=`/`project=`/`name=`. On Windows 8 dataloader workers per client is a hang risk; unnamed runs pile into `runs/detect/train`, `train2`, … at 22 MB each per client per round. |
| `task.py:209` `validate_data_structure` | Checks `batch_N/<split>/images`, but the real layout is `batch_N/images/<split>`. Always False. Dead code today — nothing calls it — so delete it or fix it, don't leave it. |
| `docs/RUNNING.md:31`, `docs/ARCHITECTURE.md:96` | Reference `task.update_data_yaml_paths()`, which no longer exists (now `materialize_data_yaml`). |
| `docs/ARCHITECTURE.md:92` | Still claims `num_examples` is a hardcoded `10`; that was fixed in PR #22. |
| `client_app.py:20` | Imports `load_yolo_model`, never uses it. `get_set_model.load_yolo_model` also defaults to `yolo8n.yaml` + `yolov8s.pt` — mismatched scales, so it would fail if anything did call it. |
| `server_app.py` `_get_unused_batch_id` | `client_to_batch_id` is never cleared, so a client keeps its shard for the whole run (correct for FL data locality) — but `used_batch_ids` is reset each round without re-adding the cached ones, so a late-joining client can be handed a shard another client already holds. |
| CI | Still no end-to-end simulation smoke test (already §10.1 of the engineering notes). Once Phase 4 passes, the `--limit 200` smoke shard from Phase 2 is exactly the fixture that makes one cheap — or lift the 450-image fixture off `origin/laptop_copy`. |
| `README.md:88` | Points at a Google Drive folder for the preprocessed dataset that no longer resolves. Replace with the Berkeley/Kaggle instructions from Phase 2 so the next person doesn't chase a dead link. |

---

## 6. Results — 2026-08-05

Run: `flwr run . --stream --run-config "num_server_rounds=2 local_epochs=20 min_clients=2 fraction_fit=1.0"`

```powershell
cd my-project
$env:PATH = "C:\Users\PRANAS\venvs\fl_yolov8\Scripts;$env:PATH"
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"   # see the note below — without this it runs on CPU
$env:FL_AV_DATA_ROOT = "<repo>\my-project"
```

### Phase 1 — the probe

```
head nc before: 80          -> after: 13          B5 REAL
same module object after train: False             B4 REAL
ORPHAN UNCHANGED: True                            <- the client returned exactly the weights it was sent
```

`ORPHAN UNCHANGED: True` is the whole of B4 in one line: the module `__init__`
pinned never moved during training, so every round shipped the global weights
straight back and FedAvg averaged its own input.

### Phase 3 — single client

`AMP: checks passed`, 5 epochs, mAP50 **0.354** on the 10-image fixture shard.

### Phase 4 — federation, all four criteria

| # | Criterion | Result |
|---|---|---|
| 1 | global checksum changes between rounds | `619.8819` → `612.2892` ✅ |
| 2 | client sent ≠ received | `698.6961` received → `679.3250` sent ✅ |
| 3 | `metrics.csv` fit + evaluate rows, non-zero mAP50 | 4 rows, mAP50 `0.0479`–`0.0785` ✅ |
| 4 | checkpoints exist and reload | `global_round_{1,2}.pt` + `global_last.pt`; reload → val mAP50 `0.0309` ✅ |

Clients trained **distinct** shards (`batch_id=4` and `batch_id=10`) on `cuda:0`.
Wall clock **55 s** vs **308 s** for the same run on CPU — a 5.5× gap, which is the
most direct evidence the GPU is actually in the loop.

mAP is meaningless at this scale (10 train images per shard). These criteria test
**plumbing**, not accuracy.

### The one that will bite the next person

flwr 1.33 builds an **isolated runtime env** per run (`~/.flwr/runtime-envs/…`) via
`uv sync`, installing this project's dependencies fresh. `torch` from PyPI on Windows
is the **CPU-only** wheel, so the carefully installed cu128 build never reaches the
Ray actors and every client silently trains on CPU — while `nvidia-smi` sits idle and
nothing errors. Setting `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` makes the
apps run in the launching environment instead.

Two related traps: `flwr run` **rewrites `pyproject.toml`** on first use, commenting
out the whole `[tool.flwr.federations]` block and migrating it to
`~/.flwr/config.toml` — edit that file afterwards, not pyproject. And the SuperLink is
detached and persistent, so it caches the CWD of whichever run started it; always
launch from `my-project/`, and `Stop-Process` it after changing directory or config.

## Cost estimate

| Phase | Wall clock | Notes |
|-------|-----------|-------|
| 1 | 15 min | mostly the ~3 GB torch cu128 download |
| 2 fast / full | 20 min / 1–2 h | 1 GB vs 6.3 GB, plus unzip |
| 3 | 10 min | 1 epoch, 200 images |
| 4 smoke | 30 min | 2 rounds × 2 clients × 200 images |
| 4 full | 6–12 h | 5 rounds × 3 clients × 6 308 images, serialised on one card |
