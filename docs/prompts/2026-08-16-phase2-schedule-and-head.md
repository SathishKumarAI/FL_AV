# Phase 2 — the schedule and the head ⚠

**Date:** 2026-08-16 · **Phase:** 2 of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 27, 28, 29, 30

⚠ **This whole prompt changes `my-project/`.** Own branch, no pipeline changes mixed in.

## Goal

Raise holdout mAP at a *fixed* budget of image-visits by fixing three things that are
currently spending gradients badly: warmup eats three of every four local epochs, the LR
schedule restarts from scratch every round so the fleet never anneals, and the 13-class
head starts from random weights. The reference to beat is **0.4173 mAP50 / 0.2313
mAP50-95** on the 1 000-image holdout at 6 rounds × 4 local epochs × 6 vehicles × 1 400
images, seed 0, condition-partitioned, against a centralised ceiling of 0.4936.

## The three facts this is built on

Checked against ultralytics **8.4.115**, not assumed:

```
warmup_epochs = 3.0   lr0 = 0.01   lrf = 0.01   cos_lr = False   freeze = None
close_mosaic  = 10    mosaic = 1.0   patience = 100   nbs = 64
```

1. **Three of every four epochs are warmup.** `local_epochs = 4` against
   `warmup_epochs = 3.0`. The Metrics tab shows box, cls **and** dfl rising across the
   four epochs of every round — each client ends the round worse than the aggregate it
   started from, and FedAvg averages six models that each went slightly backwards.
2. **The schedule restarts every round.** `lrf = 0.01` decays to 1 % of `lr0` *within*
   the round; the next round calls `YOLO.train()` fresh and starts at `lr0` again. Six
   rounds is six independent anneals. The fleet never gets the low-LR consolidation that
   makes the last epochs of a centralised run count — a structural explanation for part
   of the 15 % gap to the ceiling that has nothing to do with federation.
3. **The 13-class head is random at round 1.** COCO weights transfer the backbone and
   discard the head, so round 1 backpropagates head noise into good features.

## Deliverables

| # | Change | File |
|---|---|---|
| 1 | **Server-driven LR.** The client already receives the round number; use it. `lr0_round = lr0 · f(round / total_rounds)` with `f` a linear or cosine decay across the *whole federation*, and `lrf = 1.0` within the round so the round itself does not anneal | `client_app.py`, `server_app.py` |
| 2 | **Warmup only where it belongs.** `warmup_epochs ≈ 0.1` for every round after the first; keep the default for round 1, where the head genuinely is new | `client_app.py` |
| 3 | **`freeze=10` for round 1 only** — protect the COCO backbone while the random head settles | `client_app.py` |
| 4 | **Warm-start the head from COCO.** Copy the matching output channels out of the COCO detection head for the classes BDD100K shares with it — `person`, `car`, `bus`, `truck`, `train`, `motorcycle`, `bicycle`, `traffic light`, `stop sign` — instead of initialising them randomly. Log which classes were matched and which stayed random | `task.py` |
| 5 | **Report mAP50-95 wherever mAP50 is reported** (backlog 28) | `client_app.py`, `pipeline/report.py` |
| 6 | **Early stopping on the holdout, server-side** (backlog 29). `patience=100` inside a 4-epoch round does nothing; the decision belongs between rounds, on the shared holdout curve | `pipeline/holdout.py`, `pipeline/stages.py` |
| 7 | Every one of 1–4 behind a run-config flag defaulting to **today's behaviour**, so each is measurable alone and the reference run stays reproducible | `pyproject.toml` |

## Hard constraints

- **One knob per run.** Four changes landed together produce one number and no knowledge.
  The ablation table is the deliverable, not the final mAP.
- **Do not change** the model architecture, the image size, the batch heuristic, the
  partitioning, the seed, or the number of image-visits. Anything that changes the budget
  invalidates the comparison against 0.4173 and against the ceiling.
- **The aggregate checksum must still move every round.** Freezing the backbone and
  shrinking the LR are exactly the changes that can quietly produce a fleet that learns
  nothing while every metric still looks plausible. This project has shipped that failure
  four separate times — see `CLAUDE.md`.
- Head warm-start must **fail loudly** if the COCO head's shape does not match what it
  expects. Silently falling back to random init would make the ablation report a
  difference that was never applied.
- `pipeline/` is not touched by this branch except for deliverables 5 and 6, and those
  two go in a separate commit.

## Definition of done

```bash
python -m pytest my-project/tests -q      # 31 tests + new ones, exit 0
python -m pytest pipeline/tests -q        # 59 tests, exit 0
python -m pipeline.runner --all --yes --rounds 6 --local-epochs 4 --per-vehicle 1400
python -m pipeline.holdout --evaluate
python -m pipeline.verify
```

An ablation table in the commit body and in `docs/ML_PLAN.md`, every row at identical
image-visits, seed 0, same fleet fingerprint:

| arm | holdout mAP50 | mAP50-95 | Δ vs 0.4173 | inside the seed spread? |
|---|---|---|---|---|
| reference | 0.4173 | 0.2313 | — | — |
| + global LR schedule | | | | |
| + short warmup | | | | |
| + freeze round 1 | | | | |
| + COCO head warm start | | | | |
| all four | | | | |

Plus: the per-epoch box/cls/dfl curves from the Metrics tab showing losses **falling**
within a round rather than rising. That is the qualitative claim this phase makes, and
it must be visible, not argued.

## Out of scope

Aggregation strategies (phase 5), model architecture (backlog 34, later), augmentation
tuning (backlog 35), per-class metrics beyond reporting mAP50-95, any UI work beyond what
the Metrics tab already draws.
