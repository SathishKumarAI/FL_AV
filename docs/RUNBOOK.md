# Runbook — run it yourself, get results, compare them

No assistant required. Every command here is copy-pasteable and every one of them
prints what it did.

## 0. One-time setup (~15 min, plus a 7.6 GB download)

```powershell
# Windows. python.org 3.12, NOT conda -- Smart App Control blocks conda-forge's
# _bz2.pyd on every version tested. See docs/ENV_WINDOWS.md.
py -3.12 -m venv $env:USERPROFILE\venvs\fl_yolov8
& $env:USERPROFILE\venvs\fl_yolov8\Scripts\pip install -r my-project\requirements.txt
& $env:USERPROFILE\venvs\fl_yolov8\Scripts\pip install -r pipeline\requirements.txt
# Blackwell (RTX 50xx, sm_120) needs cu128; cu118 has no kernel for it.
& $env:USERPROFILE\venvs\fl_yolov8\Scripts\pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128
```

```bash
# Linux/macOS
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r my-project/requirements.txt -r pipeline/requirements.txt
```

The dataset needs **no Kaggle account and no token**: `kagglehub` downloads
`solesensei/solesensei_bdd100k` anonymously the first time the `dataset` stage runs.

## 1. The whole thing, one command

```powershell
.\scripts\run_pipeline.ps1                       # demo profile: ~10 min on a 5070 Ti
.\scripts\run_pipeline.ps1 -Profile full -Rounds 6 -Epochs 4 -PerVehicle 1400 -Baseline
```

```bash
./scripts/run_pipeline.sh
PROFILE=full ROUNDS=6 EPOCHS=4 PER_VEHICLE=1400 BASELINE=1 ./scripts/run_pipeline.sh
```

It runs tests → holdout → dataset → shards → fleet → validate → federate →
evaluate → verify → compare, and stops at the first failure rather than continuing.

## 2. Or drive it yourself, stage by stage

```bash
python -m pipeline.runner --list                 # what would run, what would skip
python -m pipeline.holdout --build --size 1000   # before the fleet, always
python -m pipeline.runner --all --profile demo --vehicles 6 --rounds 2 --yes
python -m pipeline.holdout --evaluate            # the honest number, per round
python -m pipeline.verify                        # the four pass criteria
python -m pipeline.baseline --rounds 2 --local-epochs 1   # the ceiling
```

Or from the browser: `python -m pipeline.server` → <http://127.0.0.1:8800>.
Control tab configures and launches; Live tab watches it.

## 3. Compare — four ways, one command each

Every arm runs the full chain and ends with a score on the **same** holdout, which is
the only number comparable between runs.

```bash
# Does the seed explain the difference? (One run is an anecdote.)
python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes

# Does the aggregation strategy matter?
python -m pipeline.experiment --preset strategies --strategies fedavg,fedadam,fedavgm --yes

# Does non-IID partitioning matter?
python -m pipeline.experiment --preset partitions --partitions condition,random,dirichlet --yes

# How much does the Dirichlet skew knob move the result?
python -m pipeline.experiment --preset alpha --alphas 0.05,0.5,100 --yes

# Does it matter that real vehicles hold unequal AMOUNTS of data?
# Every arm makes the same image-visits; only the distribution across vehicles moves,
# which is what FedAvg weights by.
python -m pipeline.experiment --preset skew --skews 0,0.8,1.5 --yes
```

Anything else, as JSON:

```bash
echo '[{"strategy":"fedadam","rounds":4},{"strategy":"fedavg","rounds":4}]' > arms.json
python -m pipeline.experiment --arms arms.json --yes
```

Each writes `pipeline/.state/experiments/<timestamp>.md` — a Markdown table, holdout
number first. To compare runs you already have:

```bash
python -m pipeline.compare --last 10          # console table
python -m pipeline.compare --last 10 --md     # paste into a doc
```

`compare` refuses to be quietly misleading: it warns when the runs differ in more
than one setting, and when they used **different fleets** (same config does not mean
same images — the fleet fingerprint proves it).

## 4. What you get

| Artifact | Where |
|---|---|
| Run report, self-contained HTML | `pipeline/reports/<timestamp>/report.html` |
| Same content, diffable | `report.md`, `report.json` |
| Honest global curve | `pipeline/.state/holdout_metrics.json` |
| Centralised ceiling | `pipeline/.state/baseline.json` |
| Experiment tables | `pipeline/.state/experiments/*.md` |
| Live dashboards | `python -m pipeline.server` |

## 5. Reading the result — in order of what matters

1. **Aggregate checksum moves every round.** Equal consecutive values mean nothing is
   being learned, whatever the metrics say. `verify` asserts it; the dashboard's
   heartbeat panel draws it.
2. **Holdout mAP50**, not the per-client number. Clients score themselves on their own
   conditions, so their average compares distributions as much as models.
3. **The gap to the centralised ceiling.** Federated learning is a trade; the size of
   the trade is the result.
4. Per-vehicle divergence — spread is expected and is what makes it federated.

Reference numbers on an RTX 5070 Ti, 6 vehicles × 1 400 images, condition-partitioned,
6 rounds × 4 local epochs, seed 0:

| | value |
|---|---|
| holdout mAP50 (round 1 → 6) | 0.3329 → **0.4173** |
| holdout mAP50-95 | 0.2313 |
| centralised ceiling, same 201 600 image-visits | 0.4936 mAP50 / 0.2770 mAP50-95 |
| retained | **84.5 %** of the ceiling (gap 0.0763) |
| per-client self-evaluated | 0.4481 — higher, because each client scores itself on its own conditions |
| wall clock | 3 296 s federated, plus about as much again for the ceiling |
| GPU energy | 82.2 Wh |
| peak VRAM | 5 087 MiB of 16 303 |

Run-to-run variance is at least ±0.016 mAP50 (an earlier ceiling with 1.667× the data
scored *lower*), so treat any difference smaller than that as noise until
`--preset seeds` says otherwise.

## 6. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Every client trains on CPU, 5.5× slower, no error | flwr built its own runtime env with the CPU-only torch wheel | `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` (the scripts set it) |
| `flwr run` exits 0 but nothing trained | it prints `Simulation Runtime crashed` and still returns success | the runner scans output for crash markers; read the stage's tail |
| Federation hangs waiting for clients | `num-supernodes` below the vehicle count | the runner overrides it on the CLI; do not edit pyproject |
| `pyproject.toml` shows a "CONFIGURATION MIGRATION NOTICE" | `flwr run` rewrote it | the runner restores it after every run; never commit that form |
| Second run reuses a stale fleet | shards match the config on disk | `python -m pipeline.validate`, then rebuild if it complains |
| `_bz2` import error on a fresh conda env | Smart App Control blocks conda-forge's `_bz2.pyd` | use a python.org 3.12 venv |
| No kernel image for the GPU | torch cu118 on Blackwell | install cu128 |
| Holdout images found inside shards | the fleet predates the holdout | rebuild the fleet; the fleet check now forces it |

## 7. Cost, before you start

| Profile | Images/vehicle | 6 vehicles × 2 rounds | 6 × 6 rounds × 4 epochs |
|---|---|---|---|
| demo | 300 | ~10 min | ~50 min |
| full | 1 400 | ~20 min | ~55 min, 82 Wh |
| full | 6 308 | hours | hours; peak VRAM 15.9 GB of 16.3 |

## 8. Packing clients onto the card — `--gpu-fraction`

Vehicles train **serialised** by default (`--gpu-fraction 1.0`). `pipeline.profile`
measured 72 client episodes of the six-round run never overlapping, with 99.1 % of the
wall clock inside a client, so this is the lever with the largest ceiling. It changes
nothing mathematically: clients are independent within a round.

```bash
python -m pipeline.runner --stages federate --profile demo --gpu-fraction 0.5 --yes
```

Measured on the RTX 5070 Ti (16 303 MiB), demo profile, 6 vehicles × 2 rounds × 1
epoch, **same fleet, built once and reused**:

| `--gpu-fraction` | clients at once | federate stage | run wall clock | peak VRAM | energy |
|---|---|---|---|---|---|
| 1.0 (default) | 1 | 215.2 s | 176 s | 7 319 MiB | 3.13 Wh |
| 0.5 | 2 | **148.3 s** | **117 s** | **15 679 MiB** | 2.65 Wh |

**1.45× on the stage, 1.50× on the federation, and the aggregate checksum still moved
every round** — 630.02 → 616.87, all four criteria green.

**The ceiling is VRAM, and it is closer than it looks.** Two demo clients already take
15 679 MiB of 16 303: `--gpu-fraction 0.33` does not fit at this profile, and the plan's
expectation of 2–2.5× was optimistic by exactly that much. The fraction is a *scheduling*
share — Ray runs ⌊1/fraction⌋ clients and caps none of them — so pick it from measured
peak VRAM, not from the client count you would like:

| Images/vehicle | Peak VRAM, one client | Safe fraction |
|---|---|---|
| 300 (demo, 320 px) | 7 319 MiB | 0.5 |
| 1 400 (full, 640 px) | 5 087 MiB | 0.5, probably 0.33 — unmeasured |
| 6 308 (full, 640 px) | 15 900 MiB | 1.0 only |

The 1 400-image row peaks *lower* than the 300-image one because batch size is chosen
per run, not per image count. Measure before trusting a row you have not run.
