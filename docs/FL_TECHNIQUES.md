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

`server_app.CustomBatchStrategy` extends `FedAvg` and adds two things Flower does not:
shard assignment per client, and checkpoint/metrics persistence. `strategy = "fedavg" |
"fedprox"` in `pyproject.toml` selects between plain averaging and a **weight-space
approximation** of FedProx — the proximal pull is applied on the client after training
(`w ← w − μ(w − w_global)`) rather than as a per-step loss term, because Ultralytics'
high-level trainer exposes no per-step hook.

That approximation is honest but it is *not* FedProx. A real comparison needs the true
proximal term or an explicit note that it is an approximation.

## The problem with adding more

`CustomBatchStrategy` inherits from `FedAvg` **by name**. Trying `FedAdam` today means
editing the class definition — so every strategy is a source change, and two strategies
cannot coexist. That is the thing to fix before running rung 5 of the ML plan.

## Proposed design: strategy as a plugin

Keep the two project-specific behaviours (shard assignment, persistence) in a **mixin**,
and compose it with whichever Flower strategy is requested:

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

Three properties this buys:

1. **Every Flower strategy works immediately**, including ones released later.
2. **The mixin is testable on its own** — assignment logic no longer entangled with
   aggregation. The `configure_fit` bug that gave every client the same shard would
   have been caught by a mixin test in isolation.
3. **DP wrappers compose**, because they wrap a strategy instance rather than subclass it.

Cost: it changes `my-project/my_project/server_app.py`, which the `pipeline/` component
is forbidden to touch. This is a **my-project change**, on its own branch, with its own
prompt — not something the pipeline should reach into.

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
a target mAP, per-vehicle divergence spread, wall clock, and GPU energy. The pipeline
already collects the last two.
