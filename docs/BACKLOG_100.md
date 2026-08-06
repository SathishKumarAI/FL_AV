# Backlog — 100 features

Ordered within each section by value per unit of effort. **P1** = do next, **P2** =
soon, **P3** = when the obvious work runs out. Anything marked ⚠ changes `my-project/`
and therefore needs its own branch and prompt.

---

## A. UI — from "ASCII log" to premium (1–24)

The current dashboard is functional and looks like tooling. These are what make it look
like a product.

| # | Feature | P |
|---|---|---|
| 1 | Design system pass: type scale, 8px spacing grid, consistent radii, one accent — replace ad-hoc inline styles | P1 |
| 2 | Real chart component with axes, gridlines, ticks and hover tooltips (still inline SVG, still no CDN) | P1 |
| 3 | Vehicle cards as a proper fleet grid: condition icon, sparkline, delta chip, status ring | P1 |
| 4 | Skeleton loaders instead of "—" while the first poll lands | P1 |
| 5 | Empty states that say what to do next, not just that data is missing | P1 |
| 6 | Toast notifications for stage transitions and failures | P2 |
| 7 | Animated round transitions so progress is felt, not read | P2 |
| 8 | Dark/light toggle that persists (currently follows OS only) | P2 |
| 9 | Keyboard shortcuts: `r` run, `s` stop, `1..9` select vehicle, `?` help | P2 |
| 10 | Command palette (⌘K) for stages, vehicles, reports | P3 |
| 11 | Responsive layout that works at 1280px and on a tablet | P2 |
| 12 | Per-vehicle detail drawer: full curve, config, shard composition, sample images | P1 |
| 13 | Sample image strip per vehicle — see what "rain / fog" actually looks like | P1 |
| 14 | Live map/schematic of the fleet with vehicles lighting up as they train | P3 |
| 15 | Round timeline scrubber: drag back through the run's history | P2 |
| 16 | Compare two runs side by side in the browser | P2 |
| 17 | Confusion-matrix / per-class mAP panel | P2 |
| 18 | Weight-flow sankey: global → clients → aggregate, widths by contribution | P3 |
| 19 | GPU panel: temperature, clocks, fan, and a cost estimate in local currency | P3 |
| 20 | Log viewer: level filter, search, regex, jump-to-error, copy-line | P2 |
| 21 | Progressive disclosure — collapse advanced panels by default | P2 |
| 22 | Accessibility pass: focus rings, ARIA on charts, contrast ≥ 4.5:1, reduced-motion | P1 |
| 23 | Export any chart as PNG/SVG | P3 |
| 24 | Server-side render of the first paint so the page is useful before JS runs | P3 |

## B. ML research (25–46)

| # | Feature | P |
|---|---|---|
| 25 | **Centralised baseline run** — FL numbers mean nothing without the ceiling | P1 |
| 26 | **Shared holdout set** no vehicle trains on, for honest global evaluation | P1 |
| 27 | Freeze the backbone for round 1 while the random 13-class head settles ⚠ | P1 |
| 28 | Report mAP50-95 everywhere mAP50 is reported | P1 |
| 29 | Early stopping on the global model, on the shared holdout | P2 |
| 30 | LR schedule tuned for short local rounds (warmup is a third of a 4-epoch round) ⚠ | P1 |
| 31 | `(rounds, local_epochs)` sweep at constant product — measures client drift | P1 |
| 32 | Per-class metrics: which of the 13 classes the fleet is failing | P2 |
| 33 | Per-condition evaluation matrix: every vehicle's model on every condition | P2 |
| 34 | Model comparison at fixed epochs: yolov8s vs 8m vs yolo11s | P2 |
| 35 | Augmentation study — mosaic helps big shards, hurts small ones | P2 |
| 36 | Class-imbalance handling; BDD100K is dominated by `car` | P2 |
| 37 | Personalisation: keep a per-vehicle head, share the backbone | P3 |
| 38 | Knowledge distillation from the centralised model into the federated one | P3 |
| 39 | Checkpoint averaging / SWA across rounds | P3 |
| 40 | Confidence calibration of the global model | P3 |
| 41 | Failure-case gallery: worst predictions per condition | P2 |
| 42 | Statistical significance across seeds — one run is an anecdote | P1 |
| 43 | Learning-curve extrapolation to predict the value of more rounds | P3 |
| 44 | Data-quality audit: mislabelled / empty-label images per shard | P2 |
| 45 | Active learning: pick the next images for a vehicle by uncertainty | P3 |
| 46 | Continual learning: new condition arrives without forgetting the old | P3 |

## C. Federated learning techniques (47–62)

See [`FL_TECHNIQUES.md`](FL_TECHNIQUES.md) — Flower already ships 24 strategies.

| # | Feature | P |
|---|---|---|
| 47 | **Strategy plugin architecture** — mixin + registry, so any Flower strategy works ⚠ | P1 |
| 48 | True FedProx proximal term rather than the weight-space approximation ⚠ | P1 |
| 49 | FedAdam / FedYogi / FedAdagrad comparison at fixed everything else | P1 |
| 50 | FedAvgM (server momentum) — cheap, often free gains on non-IID | P2 |
| 51 | Strategy leaderboard in the UI and the report | P2 |
| 52 | μ sweep for FedProx {0.001, 0.01, 0.1, 1.0} | P2 |
| 53 | Simulated faulty vehicle (label noise / random weights) — a prerequisite for robustness claims | P2 |
| 54 | Krum / Bulyan / trimmed-mean robustness runs against #53 | P2 |
| 55 | Differential privacy wrappers with a stated ε | P2 |
| 56 | Client sampling: `fraction_fit < 1.0` with many vehicles | P2 |
| 57 | Stragglers and dropouts — vehicles that miss rounds | P2 |
| 58 | Asynchronous FL | P3 |
| 59 | Hierarchical FL: vehicles → regional aggregator → global | P3 |
| 60 | Communication-efficiency: quantise / sparsify updates, measure the mAP cost | P2 |
| 61 | Secure aggregation | P3 |
| 62 | Gradient-leakage demonstration — shows *why* DP is needed | P3 |

## D. Data engineering (63–79)

| # | Feature | P |
|---|---|---|
| 63 | **Partition strategies as plugins**: condition, random, mixed, dirichlet(α), by-size, by-class | P1 |
| 64 | Dirichlet partitioning with α — the standard non-IID knob in FL papers | P1 |
| 65 | Manifest per fleet: exact image list, hashes, seed, partition — reproducibility | P1 |
| 66 | Shard validation: every image has a label, no duplicates, no cross-shard leakage | P1 |
| 67 | Condition-supply guard: refuse a fleet whose per-vehicle size exceeds the rarest condition | P1 |
| 68 | Streaming/lazy shard materialisation for fleets larger than disk | P2 |
| 69 | Dataset versioning (DVC or a content-hash manifest) | P2 |
| 70 | Incremental populate: only link what changed | P2 |
| 71 | Parquet/arrow index of image attributes instead of a 6.7 MB JSON | P3 |
| 72 | Second dataset (nuScenes / Cityscapes) behind the same interface | P3 |
| 73 | Synthetic condition augmentation: rain/fog applied to clear images | P3 |
| 74 | Train/val/test split integrity test — no image in two splits | P1 |
| 75 | Per-shard class histogram, surfaced in the UI | P2 |
| 76 | Automatic detection of stale run artifacts (already bit us once) | P2 |
| 77 | Data card per fleet, in the ML-documentation sense | P3 |
| 78 | Image dedup by perceptual hash across shards | P3 |
| 79 | Label-quality scoring, to weight or drop bad shards | P3 |

## E. Observability & reporting (80–88)

| # | Feature | P |
|---|---|---|
| 80 | MLflow logging wired for real (module exists, nothing calls it in anger) | P1 |
| 81 | Run comparison report: N runs, one table, deltas | P1 |
| 82 | Alerting: notify on failure or on a plateau | P2 |
| 83 | Cost accounting per run — kWh, and money at a configured tariff | P2 |
| 84 | Trace/timeline view of a round: who trained when, and the gaps | P2 |
| 85 | Live per-epoch streaming, not just per-round | P2 |
| 86 | Report diffing between two runs | P2 |
| 87 | Export the whole run as a reproducible bundle (config + manifest + metrics) | P2 |
| 88 | Prometheus endpoint for external dashboards | P3 |

## F. Performance & optimisation (89–95)

| # | Feature | P |
|---|---|---|
| 89 | Measure real per-client VRAM and pack clients concurrently where they fit | P1 |
| 90 | AMP / channels-last / `torch.compile` benchmark at fixed accuracy | P2 |
| 91 | Cache the dataset scan across rounds — Ultralytics rescans every time | P2 |
| 92 | Persistent client actors so the model is not reloaded per round | P2 |
| 93 | Multi-GPU / multi-node once the fleet outgrows one card | P3 |
| 94 | Mixed-resolution training: small images early, full later | P3 |
| 95 | Profile the round to find the non-training overhead | P2 |

## G. Reliability, process, docs (96–100)

| # | Feature | P |
|---|---|---|
| 96 | CI matrix: Windows + Linux, so the CWD/path traps are caught | P1 |
| 97 | Nightly scheduled smoke run against `main` | P2 |
| 98 | Resume a run from a checkpoint after a crash | P2 |
| 99 | One-command reproduction of any past run from its report bundle | P2 |
| 100 | Architecture decision records for the choices already made (assemble-don't-build, isolation rule, partition design) | P2 |

---

## If you only do ten

**1, 2, 3, 12, 13** (the UI stops looking like a log), **25, 26** (results become
meaningful), **47** (every FL technique unlocks at once), **63/64** (partitioning becomes
research-grade), **89** (the GPU stops idling between serialised clients).
