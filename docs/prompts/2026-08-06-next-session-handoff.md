# Handoff — start here next session

Paste this as the opening message. It exists so the next session spends its tokens on
work rather than on rediscovering the state of the project.

---

## Where things are

Branch `feat/pipeline-observability`, off `fix/gpu-bringup` (PR #32), off `main`.

Working, verified:

- **The federation learns.** 6 vehicles, condition-partitioned, eval mAP50 0.275 → 0.320
  over 2 rounds. All four pass criteria green.
- **`pipeline/`** — staged runner, two dashboards, per-vehicle learning analysis, GPU
  power/energy, HTML+Markdown reports. 42 tests.
- **Data** — all 10 shards hold real BDD100K (6 308 train + 1 010 val), hardlinked to the
  kagglehub cache. Attribute index cached (79 863 images).
- **Docs** — `ML_PLAN.md`, `FL_TECHNIQUES.md`, `BACKLOG_100.md`, `CONTRIBUTING.md`,
  `pipeline/docs/ARCHITECTURE.md`, and the design spec under `docs/superpowers/specs/`.

## Read these first, in this order

1. `CLAUDE.md` — hard rules, and the list of silent failures this project has shipped
2. `docs/ML_PLAN.md` — why mAP is low, the experiment ladder, model options
3. `docs/BACKLOG_100.md` — what to build, prioritised
4. `CONTRIBUTING.md` — branch/commit/merge discipline

## Environment — do not rediscover these

```powershell
# The venv is python.org 3.12, NOT conda (Smart App Control blocks conda's _bz2.pyd)
C:\Users\PRANAS\venvs\fl_yolov8\Scripts\python.exe

cd C:\Users\PRANAS\Documents\coding\shelf\ai\federated-yolov8
python -m pipeline.server                 # dashboards on :8800
python -m pipeline.runner --list          # what would run, what would skip
```

Traps that already cost hours — all now guarded in code, listed so they are not
re-debugged:

| Trap | Guard |
|---|---|
| flwr builds its own runtime env with CPU-only torch → silent CPU training at 5.5× wall clock | `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` in `paths.subprocess_env` |
| flwr exits **0** after printing `Simulation Runtime crashed` | `stages.CRASH_MARKERS` scans output |
| The detached SuperLink caches the CWD **and env** of whichever run started it | `runner._stop_stale_superlink()` before every federate |
| Ray rejects `num_gpus` when attaching to an existing cluster | `init-args` omitted when `--ray-address` is set |
| Server assigns `batch_id` from 1..10 regardless of vehicle count | fleet always materialises all 10 shards |
| Logs scatter across `my-project/logs` and the repo root | `paths.log_dirs()` searches both |
| `flwr run` rewrites `pyproject.toml`, commenting out the federations block | `runner._restore_pyproject()`; never commit that form |
| Stale `results.csv` from an old run contaminates analysis | filtered by `fleet.json` mtime |
| A UI served by a stale server process renders blank | version-skew banner in `index.html` |

## The one number that matters

**Round-over-round aggregate checksum.** Identical consecutive values mean the
federation is not learning, whatever the metrics say. Everything else is secondary.

```bash
python -m pipeline.verify
```

## Next tasks, in order

### 1. Premium UI pass — the top ask

The dashboard works and looks like a log file. `docs/BACKLOG_100.md` items 1–5, 12, 13, 22.
Concretely: a design system (type scale, 8px grid, one accent), real charts with axes and
tooltips, fleet cards with condition icons and sparklines, a per-vehicle detail drawer
showing sample images from that vehicle's shard, skeleton loaders, and an accessibility
pass. Still inline SVG, still no CDN, still no build step.

**Load the `frontend-design` skill before starting.**

### 2. Read the long run's result

A 6-vehicle × 1 400-image × 6-round × 4-epoch condition-partitioned run was launched at
the end of the previous session (~2 h). Check `pipeline/reports/` for the newest report
and `pipeline/.state/longrun.log`. It should show markedly higher mAP than 0.320 — that
run had only 2 effective epochs, this one has 24.

### 3. Centralised baseline + shared holdout — backlog 25, 26

Federated numbers are uninterpretable without them. This is the highest-value ML work.

### 4. Strategy plugin architecture — backlog 47

⚠ Changes `my-project/server_app.py`, so it needs its own branch and its own prompt.
`CustomBatchStrategy` inherits `FedAvg` by name; a mixin + registry unlocks all 24
Flower strategies at once. Design is in `docs/FL_TECHNIQUES.md`.

### 5. Partition plugins + Dirichlet — backlog 63, 64

`condition | random | mixed` exist. Dirichlet(α) is the standard non-IID knob in the
literature and makes results comparable with published work.

## Rules that are not negotiable

1. `pipeline/` never modifies `my-project/` — enforced by a test.
2. No data, checkpoints, reports or credentials committed — enforced by a test.
3. Assemble before building: MLflow owns metrics, Ray owns actor internals. A bespoke
   dashboard was proposed once and correctly rejected.
4. Fail loudly. A stage that fails halts the chain.
5. Verify the effect, not the absence of errors. Put the number in the commit message.

## Known imperfections, stated rather than hidden

- MLflow logging is wired but nothing calls it in a real run yet (backlog 80).
- Each client evaluates on its own val split, so there is no honest global metric yet
  (backlog 26).
- The FedProx implementation is a weight-space approximation, not the true proximal
  term — fine as a baseline, not fine as a published comparison (backlog 48).
- Vehicles train serialised. At the demo profile one client peaks at ~5 GB of 16.3 GB,
  so there is real headroom to pack them concurrently (backlog 89).
- `pipeline/docs/ARCHITECTURE.md` predates the vehicle-learning and partition features;
  its component diagram needs a refresh.
