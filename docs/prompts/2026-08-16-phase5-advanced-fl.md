# Phase 5 — advanced FL, once a comparison can mean something ⚠

**Date:** 2026-08-16 · **Phase:** 5 of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 48, 49, 50, 37, 53, 54, 56, 57, 60, 55

⚠ Most deliverables change `my-project/`. Own branch per deliverable.

## Goal

Turn twelve reachable strategies into a **leaderboard whose gaps are larger than the
noise floor**. The plumbing already exists — `BatchAssignmentMixin` composes with any
Flower strategy, and an unregistered name raises rather than silently falling back to
FedAvg. What does not exist is a fair contest: the only comparison on disk ran 2 rounds
× 1 epoch, where a server-side optimiser has not yet had anything to optimise.

**Do not start this phase before phase 3 has produced the spread.** Ranking six
strategies whose deltas are inside run-to-run variance is a way to spend GPU hours
generating a ranking of noise.

## The correction this phase is built on

`docs/FL_TECHNIQUES.md` said Ultralytics exposes no per-step hook, so `fedprox` had to be
a weight-space approximation applied after training (`w ← w − μ(w − w_global)`). Checked
against **ultralytics 8.4.115**, that is half wrong, and the wrong half is the useful one:

- `"optimizer_step"` **is** a key in
  `ultralytics.utils.callbacks.base.default_callbacks`, but `BaseTrainer.optimizer_step`
  never calls `run_callbacks("optimizer_step")`. **The callback is dead** — registering
  one buys nothing and looks like it works. This is exactly the shape of silent no-op
  this project keeps shipping; do not use it.
- `BaseTrainer.optimizer_step` **is** an ordinary method, called from `_do_train` after
  `scaler.scale(loss).backward()` and before `optimizer.zero_grad()`, and `YOLO.train()`
  accepts a `trainer=` class.

So a real per-step proximal term is a `DetectionTrainer` subclass that adds
`μ · (w − w_global)` to `p.grad`, then calls `super().optimizer_step()`, with the global
weights captured at round start. About fifteen lines.

## Deliverables, in order of expected value on this dataset

| # | Item | Notes |
|---|---|---|
| 1 | **True FedProx** ⚠ — trainer subclass as above. Keep the current approximation reachable as `fedprox-approx` so the two can be compared; that comparison is itself a result | `client_app.py`, `task.py` |
| 2 | **μ sweep** {0.001, 0.01, 0.1, 1.0} once #1 is real. Too large freezes clients to the global model — the run that produces no learning should be *visible*, via the checksum, not mysterious | `pipeline/experiment.py` preset |
| 3 | **Personalised heads** ⚠ (backlog 37) — share the backbone, keep a per-vehicle 13-class head that is never aggregated. Condition-partitioned vehicles is the textbook case, and it attacks the per-vehicle spread directly (0.4069 worst, 0.4754 best) | `server_app.py`, `client_app.py` |
| 4 | **FedAvgM** — server momentum, one keyword argument, often a free win on non-IID | registry only |
| 5 | **FedAdam / FedYogi at real budget** — 6 rounds × 4 epochs, not 2 × 1 | runs only |
| 6 | **Simulated faulty vehicle** (backlog 53) — label noise, or returning random weights. A prerequisite: without it there is nothing for Krum to be robust *against* | `pipeline/vehicles.py`, `client_app.py` ⚠ |
| 7 | **Krum / Bulyan / trimmed-mean against #6** (backlog 54) — the robustness result | runs only |
| 8 | **Client sampling and stragglers** (56, 57) — `fraction_fit < 1.0`, vehicles that miss rounds. This is what a real fleet does | `server_app.py` ⚠ |
| 9 | **Communication cost** (backlog 60) — quantise or sparsify updates, report bytes per round **and** the mAP paid for them. The only axis where FL has an engineering story beyond privacy | both ⚠ |
| 10 | **DP wrappers with a stated ε** (backlog 55) — they wrap an instance, so they compose with the mixin. FL alone is not privacy: gradients leak, which is the point of backlog 62 | `server_app.py` ⚠ |

## Hard constraints

- **Fixed everything else.** Same seed, same fleet hash, same rounds × local epochs, same
  model, same schedule. Two changes at once produces one number and no knowledge.
- **A strategy that produces a static aggregate checksum has failed**, whatever its
  metrics say. Assert it per arm, not by eye. Large μ and heavy clipping are precisely
  the settings that yield a plausible-looking run in which nothing moves.
- **Every arm is scored on the shared holdout.** Before it existed, a strategy comparison
  would partly have ranked whoever drew the easier conditions.
- **Do not claim robustness or privacy without the adversary.** Krum without a faulty
  vehicle, and DP without a stated ε, are both decoration.
- The pipeline mirrors the strategy name list in `pipeline/stages.py` rather than
  importing it, and `my-project/tests/test_strategy_registry.py` asserts the two have not
  drifted. Adding a name means updating both, in the same commit.

## Definition of done

```bash
python -m pytest my-project/tests -q
python -m pytest pipeline/tests -q
python -m pipeline.experiment --preset strategies \
    --strategies fedavg,fedprox,fedavgm,fedadam,fedyogi \
    --profile full --rounds 6 --epochs 4 --yes
python -m pipeline.compare --last 5
```

A leaderboard, in the commit and in `docs/FL_TECHNIQUES.md`, with per strategy: final
holdout mAP50 and mAP50-95, rounds to reach a target mAP, per-vehicle divergence spread,
wall clock, GPU energy — and a **± column from phase 3**, with any strategy inside the
spread of FedAvg reported as *no measured difference*, not as a placing.

## Out of scope

Asynchronous FL (58), hierarchical aggregation (59), secure aggregation (61), the
gradient-leakage demonstration (62), and knowledge distillation from the centralised
model (38). All are phase-6 material and all are cheaper to justify once the leaderboard
shows which direction is worth the hours.
