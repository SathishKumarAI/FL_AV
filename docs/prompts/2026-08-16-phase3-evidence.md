# Phase 3 — evidence: the noise floor, before any claim of "better"

**Date:** 2026-08-16 · **Phase:** 3 of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 42, 31, 81, 28

## Goal

Produce the number that every other comparison in this repo is measured against: the
run-to-run spread at fixed configuration. Until it is known, no difference between two
approaches means anything, and this project has already been bitten by that.

**The evidence it has been bitten.** A centralised ceiling trained on 14 000 images for
24 epochs — 336 000 image-visits, **1.667× the budget** — scored **0.4771**, *lower* than
the 8 400-image ceiling's **0.4936**. More data and more compute produced a worse model.
Whatever the cause, it puts a floor under believable differences at roughly ±0.016
mAP50 — larger than several of the deltas this project might otherwise want to call
results.

This phase ships no feature. It ships the right to use the word "better".

## Hard constraints

- **Change exactly one setting per arm.** `pipeline/experiment.py` already enforces this
  by construction; do not add an arms file that varies two things.
- **Prove data identity, do not assert it.** Every arm records the fleet fingerprint;
  arms that claim to share data must share the hash. Phase 4 makes that hash content-based
  — until then, quote the fingerprint that exists.
- **Every arm is scored on the shared holdout**, never on clients' own val splits. The
  self-evaluated number is 0.031 higher and that gap is flattery, measured.
- **No arm is run at demo budget and reported as a result.** The existing fedavg-vs-fedadam
  comparison ran 2 rounds × 1 epoch and produced 0.0042 vs 0.0000: real machinery, no
  information. Demo budget is for testing the harness, not the hypothesis.
- Report **mAP50 and mAP50-95** for every arm. mAP50 alone flatters a detector with sloppy boxes.

## Inputs

- `pipeline/experiment.py` — presets `seeds`, `strategies`, `partitions`, `alpha`
- `pipeline/compare.py` — run-to-run comparison
- The Metrics tab already groups repeats and shows their spread; it needs runs to group
- Reference: 0.4173 mAP50 / 0.2313 mAP50-95, 6 rounds × 4 epochs × 6 vehicles × 1 400 images

## Deliverables

| # | Item | Command / file |
|---|---|---|
| 1 | **Seed repeats, at real budget** — the phase's whole point | `python -m pipeline.experiment --preset seeds --seeds 0,1,2 --profile full --rounds 6 --epochs 4 --yes` |
| 2 | **`(rounds, local_epochs)` at constant product** — 12×2, 6×4, 3×8, identical image-visits. Measures client drift directly; phase 2 predicts fewer local epochs win | new preset `budget` in `experiment.py` |
| 3 | **IID control** — the same run with `--partition random`. How much of the result is federation and how much is just data | `--preset partitions` |
| 4 | **A ± column everywhere.** `compare.py` and `report.py` print mean ± spread over repeats, and mark any delta inside the spread as *no measured difference* rather than as a winner | `pipeline/compare.py`, `pipeline/report.py` |
| 5 | **A significance note in `docs/ML_PLAN.md`** stating the measured spread, the date it was measured, and the configuration it applies to | `docs/ML_PLAN.md` |
| 6 | **MLflow actually recording it** (backlog 80). It refused the last run: *"the filesystem tracking backend is in maintenance mode"*. It needs a SQLite backing store — `mlflow-tracking-uri sqlite:///…` — not just a call to the sink | `pipeline/mlflow_sink.py`, `pipeline/paths.py` |

## Definition of done

```bash
python -m pytest pipeline/tests -q
python -m pipeline.experiment --preset seeds --seeds 0,1,2 --profile full --yes
python -m pipeline.compare --last 3
python -m pipeline.holdout --evaluate
```

In the commit body, and in `docs/ML_PLAN.md`:

- three holdout mAP50 values at identical configuration, their mean, and their spread
- the same for mAP50-95
- one sentence naming the threshold a future difference must exceed, e.g. *"deltas below
  X mAP50 at this configuration are not distinguishable from run-to-run variance"*
- the constant-product sweep table, with the winner marked only if it clears that threshold

## Out of scope

Any change to the model, the schedule, the data pipeline or the aggregation strategy.
This phase varies **nothing but the seed and the round/epoch split**; that is what makes
it a measurement rather than an experiment.
