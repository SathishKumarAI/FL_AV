# Prompt — a shared holdout, and the centralised ceiling to measure against

Written before the code. Backlog 25 and 26, the two items `docs/ML_PLAN.md` calls the
highest-value ML work left.

## The problem

Every reported number in this project is a client evaluating on **its own** val split.
Vehicle 3 trains on rain and is scored on rain; vehicle 1 trains on daytime city and is
scored on daytime city. So:

- The per-vehicle curves are not comparable with each other — an easier condition
  reads as a better vehicle.
- There is no global metric at all. `metrics.csv` averages numbers that were measured
  on different distributions, which is not a mean of anything.
- 0.455 mAP50 has no scale. Is that near the ceiling for this budget, or half of it?
  Nothing in the repo can answer that, so no experiment can be called a success or a
  failure.

## What to build

1. **A shared holdout** carved out of the val pool *before* the fleet is assigned, so
   no vehicle can train or self-evaluate on it. One distribution, held fixed across
   runs, deterministic from a seed.
2. **Global evaluation against it.** Score every `global_round_*.pt` checkpoint on the
   holdout and write the curve. This is the first honest global number in the project.
3. **A centralised baseline**: one model trained on the *pooled* union of every
   vehicle's training images, at a comparable budget, scored on the same holdout. That
   is the ceiling federation is trading against, and the gap is the actual result.

## How, given the constraints

`pipeline/` may not modify `my-project/`, and server-side evaluation lives in
`my-project/my_project/server_app.py`. So evaluation happens **out of band**: the
checkpoints are already written to disk, and Ultralytics can score them without the
federation knowing. No change to `my-project/` and no ⚠ branch needed.

New stages, in chain order:

| Stage | Gated | Does |
|---|---|---|
| `holdout` | no | carve and materialise the shared set, once |
| `evaluate` | no | score every global checkpoint on it, write the curve |
| `baseline` | yes | train the centralised model on pooled data, score it |

`holdout` runs **before** `fleet`, and `build_fleet` subtracts the holdout names from
the val pool. Ordering is load-bearing: a holdout carved afterwards would already be
inside somebody's val split, and the number would be quietly self-referential — the
exact failure mode this project keeps producing.

## Non-negotiable

- Deterministic: same seed, same holdout, across machines and runs.
- Not committed. Images are hardlinks into the kagglehub cache; the ignore rules and
  the test that asserts them must cover the new directories in the same commit.
- Fails loudly: evaluating a checkpoint that does not exist is an error, not a zero.
- The baseline is expensive, so it is gated like every other GPU stage.

## Verification

- `python -m pytest pipeline/tests -q`
- A holdout is disjoint from every vehicle's train and val list — asserted by test,
  and re-asserted against the real fleet on disk.
- `python -m pipeline.holdout --evaluate` prints a mAP50 per round for the finished
  6-round run, on data no vehicle ever saw.
- `python -m pipeline.baseline` trains the pooled model and prints the gap.
