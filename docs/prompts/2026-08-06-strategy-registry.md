# Prompt — a strategy registry, so every Flower strategy is reachable

Written before the code. Backlog 47. ⚠ This changes `my-project/`, so it gets its own
branch and its own prompt, per the project rules.

## The problem

`CustomBatchStrategy(FedAvg)` inherits FedAvg **by name**. Everything the project
needs from a strategy — per-client shard assignment, the aggregate checksum that is
this repo's most important signal, global checkpointing, metrics rows — is welded to
that one base class.

Flower ships 20+ strategies in this version. Trying FedAdam today means copying the
whole class and changing one word in the class statement, then maintaining two copies
of the shard-assignment logic that has already produced two silent failures (B9: one
shared `FitIns` mutated for every client; B7: checkpointing skipped every round).

## What to build

1. **`BatchAssignmentMixin`** — everything project-specific, calling `super()`
   cooperatively so it composes with any Flower strategy.
2. **A registry** mapping a name to a base strategy, built by probing what this
   Flower version actually exports, so an absent strategy is absent rather than an
   ImportError at server start.
3. **`build_strategy(name, ...)`** — composes `type(name, (Mixin, Base), {})` and
   passes each base only the keyword arguments its `__init__` actually accepts.
   FedAdam takes `eta`/`tau`, FedAvgM takes `server_momentum`, FedAvg takes neither;
   filtering by signature is what makes one call site work for all of them.
4. `strategy=` and its hyperparameters readable from `run_config`, and reachable
   from the pipeline as `--strategy`.

`CustomBatchStrategy` stays as a name — it is what the tests, the README and three
docs refer to — but becomes `BatchAssignmentMixin + FedAvg`, which is what it always
was in effect.

## Non-negotiable

- **An unknown strategy name fails loudly at server start.** Falling back to FedAvg
  would produce a run labelled FedAdam that is not one, which is the exact class of
  silent failure this repo collects.
- Every existing signal survives: `batch_id` per client from a copied config dict,
  `Aggregated parameters with checksum:`, per-round checkpoints, `metrics.csv`.
- FedProx keeps working: it is FedAvg plus `proximal_mu` shipped to the clients.
- `my-project/tests` stays green — those tests are the guard on B7 and B9.

## Verification

- `python -m pytest my-project/tests pipeline/tests -q`
- A new test builds a strategy for every registered name and asserts the mixin's
  method resolution order puts the project behaviour first.
- A test asserts an unknown name raises rather than falling back.
- A test asserts FedAdam receives `eta` while FedAvg is not handed it.
