# Getting good results — the ML plan

Written after the first real runs. It answers three questions: why mAP is low, what to
change, and in what order to spend GPU hours.

## Why mAP looks low — it is underfitting, and the data says so

The 6-vehicle demo reached eval mAP50 **0.275 → 0.320** over two rounds. That is not a
broken federation; it is a barely-trained one.

| Evidence | Reading |
|---|---|
| mAP rose every round and had not flattened | still on the steep part of the curve |
| val loss ≈ train loss (gap −0.02 to −0.08, val *below* train) | no overfitting whatsoever |
| 2 rounds × 1 local epoch = **2 effective epochs** | YOLOv8 detection heads need tens |
| 300 images/vehicle = 1 800 total, vs BDD100K's 70 000 | 2.6% of the data |
| The 13-class head is **randomly initialised** | COCO weights transfer the backbone, not the head |

`pipeline.vehicle_metrics.fit_diagnosis()` computes this rather than asserting it, and
prints `UNDERFIT — mAP is still climbing every round and the run stopped early.`

The one counter-example is instructive: a 20-epoch run on a **10-image** shard showed
train 1.45 / val 1.96, gap **+0.50** — textbook overfitting. So the pipeline can detect
both; the demo simply sat at the other end.

**Conclusion: more effective epochs on more data per vehicle.** Not a different model,
not a different strategy — those come after the baseline is trained enough to compare.

## The constraint that shapes everything: condition supply

Condition-biased partitioning is only real while the condition has images to give.

| Condition | Available |
|---|---|
| night | 31 900 |
| daytime city | 24 878 |
| highway | 19 878 |
| snow | 6 318 |
| rain / fog | 5 951 |
| dawn / dusk | 5 805 |
| **overcast residential** | **1 419** |
| parking / tunnel | 582 |

Asking for 6 308 images/vehicle exhausts every rare condition, and the shard silently
tops up with whatever is left — turning a non-IID run into a nearly-IID one while still
calling itself condition-partitioned. **Keep `--per-vehicle` below the rarest condition
in the fleet**, or drop that condition from the profile.

This is why the current run uses `--per-vehicle 1400`: it is the largest size that keeps
all six conditions genuinely distinct.

## Experiment ladder

Run in this order. Each rung answers one question and feeds the next.

| # | Run | Answers | Cost |
|---|---|---|---|
| 1 | 6 veh × 1 400 img × 6 rounds × 4 epochs, condition | Does more training fix the low mAP? (24 effective epochs) | ~2 h |
| 2 | Same, `--partition random` | **The IID control.** How much of the result is federation vs. just data? | ~2 h |
| 3 | Centralised baseline: one model on the pooled 8 400 images, 24 epochs | The ceiling. FL should approach, not beat, this | ~1.5 h |
| 4 | 3 veh × 5 000 img (drop the rare conditions), 8 rounds × 4 epochs | Does more data per vehicle beat more vehicles? | ~4 h |
| 5 | Best config × {FedAvg, FedProx, FedAdam, FedYogi} | Which aggregation suits non-IID driving data | 4 × best |
| 6 | Full 6 308 × 10 vehicles, 10 rounds × 4 epochs | The headline number | ~12 h |

**Rung 3 is the one people skip and shouldn't.** Without a centralised baseline, a
federated mAP is a number with nothing to compare it to. FedAvg on non-IID data is
*expected* to trail centralised training; the interesting quantity is the gap, and how
much each strategy closes it.

## Update 2026-08-16 — where the epochs are actually going

Rungs 1–3 are done and the fleet retains **84.5 %** of a budget-matched centralised
ceiling (0.4173 vs 0.4936 mAP50 on the shared holdout). The question changed from *"is it
underfitting?"* to *"why do 24 effective epochs only reach 0.4173?"* Three facts, checked
against the installed **ultralytics 8.4.115** rather than assumed:

```
warmup_epochs = 3.0   lr0 = 0.01   lrf = 0.01   cos_lr = False   freeze = None
close_mosaic  = 10    mosaic = 1.0   patience = 100   nbs = 64   cache = False
```

| | Fact | Consequence |
|---|---|---|
| 1 | `warmup_epochs = 3.0` inside a 4-epoch round | **three of every four epochs are warmup.** The Metrics tab shows box, cls *and* dfl rising across every round: each client ends the round worse than the aggregate it started from |
| 2 | `lrf = 0.01` decays within the round; the next round calls `train()` fresh at `lr0` | **six independent anneals, never one.** The fleet never gets the low-LR consolidation that makes a centralised run's last epochs count — a structural share of the 15 % gap that has nothing to do with federation |
| 3 | COCO weights transfer the backbone and discard the head | **round 1 backpropagates random-head noise into good features** |

Fix 1 and 2 together with a **server-driven schedule**: the round number is already
broadcast, so set `lr0_round = lr0 · f(round / total_rounds)`, `lrf = 1.0` within the
round, and `warmup_epochs ≈ 0.1` after round 1. One global anneal spread across rounds.

Fix 3 twice, and measure the two separately: `freeze=10` for round 1, **and warm-start
the 13-class head from the COCO head rows** for the classes BDD100K shares with COCO
(`person`, `car`, `bus`, `truck`, `train`, `motorcycle`, `bicycle`, `traffic light`,
`stop sign`). Starting from a detector rather than from noise is the highest
expected-value item on this page and it is not in the original backlog.

**The precision caveat that governs all of it.** A ceiling trained on 14 000 images for
24 epochs — 1.667× the budget — scored **0.4771**, *lower* than the 8 400-image ceiling's
**0.4936**. Run-to-run variance here is at least ±0.016 mAP50. Measure the spread across
seeds before believing any of the deltas above; that is
[phase 3](prompts/2026-08-16-phase3-evidence.md).

Full ordering and gates: [`docs/PHASED_PLAN.md`](PHASED_PLAN.md).

## Hyperparameters worth changing, in order of expected effect

1. **Effective epochs** (`rounds × local_epochs`). Currently the binding constraint.
   Target ≥ 24 before drawing any conclusion about a strategy.
2. **`local_epochs` vs `rounds` split.** More local epochs = fewer communication rounds
   but more client drift on non-IID data — the exact thing FedProx's proximal term
   exists to damp. Worth a dedicated sweep: (rounds, epochs) ∈ {(12,2), (6,4), (4,6)}
   at constant product.
3. **Batch size.** `get_optimal_batch_size()` returns 16 whenever >10 GB is free, which
   ignores dataset size. On small shards this interacts badly with Ultralytics' nominal
   batch of 64: below 4 batches per epoch, **no optimizer step happens at all**. The
   client warns when the arithmetic says so.
4. **LR warmup.** `warmup_epochs=3` on a shard with 88 batches/epoch means the first
   ~100 iterations are warmup — a third of a 4-epoch round spent ramping. Lower it for
   short local rounds.
5. **Augmentation.** Mosaic helps on large datasets and hurts on tiny ones. `close_mosaic=10`
   never triggers in a 4-epoch round.

## Model suggestions

Current: `yolov8s` (11.1 M params) with a 13-class head built from `yolov8s-13.yaml`.

| Option | When it is the right call |
|---|---|
| **yolov8s** (current) | Correct default: fits the 16 GB card with room, trains fast enough to iterate |
| **yolov8m** (25.9 M) | Once rung 1–3 show the small model has plateaued rather than undertrained. ~2.3× the compute; still fits |
| **yolov8n** (3.2 M) | If the goal shifts to on-vehicle inference cost. Expect a real mAP drop |
| **yolo11s / yolo26** | Ultralytics 8.4 ships newer architectures. Worth one head-to-head at fixed epochs before committing the project to v8 |
| **Freeze the backbone for round 1** | The 13-class head is random at start; freezing the backbone for the first round stops random-head gradients from disturbing good COCO features. Cheap, often worth 1–3 mAP points |

Do not change the model before rung 3. Swapping architectures while the baseline is
undertrained produces a comparison that says nothing.

## Evaluation protocol

- **Report mAP50 *and* mAP50-95.** mAP50 alone flatters a detector with sloppy boxes.
- **Evaluate the global model on a held-out set no vehicle trained on.** Currently each
  client evaluates on its own val split, which measures per-condition fit, not fleet
  generalisation. A shared holdout is the missing piece — see the backlog.
- **Watch the train/val gap per vehicle**, not just the aggregate. A single vehicle
  overfitting its rare condition is invisible in the fleet mean.
- **Fix the seed and report it.** Shard assignment is seeded; results are otherwise not
  comparable between runs.
