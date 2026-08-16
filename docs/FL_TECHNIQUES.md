# Federated learning techniques — making them swappable

The goal: try many FL algorithms without rewriting the project each time. Most of the
work is *plumbing*, not implementation — Flower 1.33 already ships 24 strategies.

## What is already available

```python
import flwr.server.strategy as st   # 24 exported
```

| Family | Strategies | Use for |
|---|---|---|
| Averaging | `FedAvg`, `FedAvgM`, `FedMedian`, `FedTrimmedAvg` | baselines; median/trimmed resist outlier clients |
| Proximal | `FedProx` | **non-IID client drift** — the obvious fit for condition-partitioned vehicles |
| Adaptive server optimisers | `FedAdam`, `FedYogi`, `FedAdagrad`, `FedOpt` | faster convergence when client updates are noisy |
| Byzantine-robust | `Krum`, `Bulyan`, `FaultTolerantFedAvg` | a vehicle sending garbage, deliberately or not |
| Differential privacy | 6 wrappers (client/server × fixed/adaptive clipping) | privacy claims that survive scrutiny |
| Fairness | `QFedAvg` | when the worst vehicle matters more than the mean |
| Gradient boosting | `FedXgb*` | not applicable here (not a tree model) |

## What this repo has today

**Built 2026-08-06 (backlog 47).** `BatchAssignmentMixin` holds the two things Flower
does not do — shard assignment per client, and checkpoint/metrics persistence — and
composes with any Flower strategy:

```bash
python -m pipeline.runner --all --strategy fedadam --yes
flwr run . --run-config 'strategy="fedyogi" eta=0.01 tau=0.001'
```

Twelve are registered in this Flower build: `fedavg`, `fedprox`, `fedadam`, `fedyogi`,
`fedadagrad`, `fedavgm`, `fedmedian`, `fedtrimmedavg`, `krum`, `bulyan`, `qfedavg`,
`faulttolerantfedavg`. Each base is passed only the keyword arguments its `__init__`
accepts, and anything dropped is logged — a silently dropped `eta` would make a FedAdam
sweep report identical numbers for every value and look like the knob does nothing.
An unregistered name raises at server start rather than falling back to FedAvg.

`strategy = "fedprox"` still selects a **weight-space approximation** of FedProx — the proximal pull is applied on the client after training
(`w ← w − μ(w − w_global)`) rather than as a per-step loss term.

That approximation is honest but it is *not* FedProx. A real comparison needs the true
proximal term or an explicit note that it is an approximation.

### Correction, 2026-08-16: the per-step hook exists

This document previously said the approximation was necessary because Ultralytics'
high-level trainer exposes no per-step hook. Checked against the installed
**ultralytics 8.4.115**, that is half wrong, and the wrong half is the useful one:

| Checked | Result |
|---|---|
| `"optimizer_step" in ultralytics.utils.callbacks.base.default_callbacks` | **True** |
| `BaseTrainer.optimizer_step` calls `run_callbacks("optimizer_step")` | **False** — the callback never fires |
| `BaseTrainer.optimizer_step` is an overridable method called from `_do_train` between `backward()` and `zero_grad()` | **True** |
| `YOLO.train()` accepts a `trainer=` class | **True** |

Two consequences, and the first one matters more:

1. **Registering an `"optimizer_step"` callback is a silent no-op.** It imports, it
   registers, it never runs, and the resulting μ sweep would report identical numbers for
   every μ and read as "the knob does nothing" — the exact failure shape catalogued in
   `CLAUDE.md`. Do not use the callback.
2. **True FedProx is about fifteen lines**: a `DetectionTrainer` subclass whose
   `optimizer_step` adds `μ · (w − w_global)` to `p.grad` and then calls `super()`, with
   the global weights captured at round start. Passed in as `model.train(trainer=...)`.

Planned in [phase 5](prompts/2026-08-16-phase5-advanced-fl.md) of
[`docs/PHASED_PLAN.md`](PHASED_PLAN.md), deliberately after the noise floor is known —
ranking strategies whose deltas sit inside run-to-run variance ranks noise.

## The design, as built

The problem it solved: `CustomBatchStrategy` inherited `FedAvg` **by name**, so trying
`FedAdam` meant copying the class — every strategy a source change, and two copies of
shard-assignment logic that had already produced two silent failures.

`CustomBatchStrategy` still exists and still means FedAvg; it is now literally
`BatchAssignmentMixin + FedAvg`. Sketch of the mechanism:

```python
class BatchAssignmentMixin:
    """Shard assignment + checkpointing. Knows nothing about aggregation."""
    def configure_fit(self, rnd, parameters, cm):
        ins = super().configure_fit(rnd, parameters, cm)      # whatever base is
        return [(c, self._with_batch_id(c, i)) for c, i in ins]

STRATEGIES = {"fedavg": FedAvg, "fedprox": FedProx, "fedadam": FedAdam,
              "fedyogi": FedYogi, "fedavgm": FedAvgM, "fedmedian": FedMedian,
              "krum": Krum, "qfedavg": QFedAvg}

def build_strategy(name: str, **kw):
    base = STRATEGIES[name]
    return type(f"Batch{base.__name__}", (BatchAssignmentMixin, base), {})(**kw)
```

Three properties this bought:

1. **Every Flower strategy works immediately**, including ones released later.
2. **The mixin is testable on its own** — assignment logic no longer entangled with
   aggregation. The `configure_fit` bug that gave every client the same shard would
   have been caught by a mixin test in isolation.
3. **DP wrappers compose**, because they wrap a strategy instance rather than subclass it.

It changed `my-project/my_project/server_app.py`, so it went on its own branch with its
own prompt (`docs/prompts/2026-08-06-strategy-registry.md`) — the pipeline never reaches
into that package. The pipeline mirrors the name list in `pipeline/stages.py` rather than
importing it, to stay free of flwr and ultralytics; `my-project/tests/test_strategy_registry.py`
asserts the two lists have not drifted.

## Per-strategy notes for this dataset

- **FedProx** — the natural first comparison. Non-IID by construction here, and μ is the
  one knob. Sweep μ ∈ {0.001, 0.01, 0.1, 1.0}; too large freezes clients to the global model.
- **FedAdam / FedYogi** — server-side adaptivity usually helps most when clients are many
  and updates noisy. With 6 clients the benefit may be small; worth measuring, not assuming.
- **FedAvgM** — server momentum. Cheap to try, often a free win on non-IID.
- **Krum / Bulyan** — only meaningful once a *faulty* vehicle is simulated. Add a
  "corrupt vehicle" mode (label noise, or returning random weights) before claiming
  robustness results; otherwise there is nothing to be robust against.
- **DP wrappers** — the honest privacy story. Note that FL alone is not a privacy
  guarantee: gradients leak. Any privacy claim needs the clipping/noise wrappers plus a
  stated ε.

## Evaluation discipline

Comparing strategies is only valid at **fixed everything else**: same seed, same shards,
same rounds × local epochs, same model. The pipeline already fixes the seed and
materialises deterministic shards, so the remaining discipline is not changing two
things at once.

Report per strategy: final global mAP50 / mAP50-95 on a shared holdout, rounds to reach
a target mAP, per-vehicle divergence spread, wall clock, and GPU energy. **The shared
holdout now exists** (`python -m pipeline.holdout --evaluate`), so this comparison is
finally possible — before it, every strategy would have been scored by each client on
its own distribution, and the winner would partly have been whoever drew the easier
conditions.

The FedAvg reference to beat, 6 rounds x 4 local epochs x 6 vehicles x 1 400 images,
condition-partitioned, seed 0: **0.4334 mAP50 / 0.2454 mAP50-95** on the 1 000-image
holdout, 3 296 s, 82.2 Wh.
