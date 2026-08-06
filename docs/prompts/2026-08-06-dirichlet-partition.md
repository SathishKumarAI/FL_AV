# Prompt — partition strategies as plugins, with Dirichlet(α)

Written before the code. Backlog 63 and 64.

## The problem

Three partitions exist — `condition`, `random`, `mixed` — and they are a chain of
`if`s inside `assign()`. Adding a fourth means editing that function, and the fleet
stage decides whether a rebuild is needed by *guessing*: it checks whether every
vehicle's label is the string `"random mix"`. That guess cannot tell a `condition`
fleet from a `mixed` one, and would be silently wrong for anything new.

Neither `condition` nor `random` is the knob the FL literature reports against.
Papers sweep **Dirichlet α**: α → 0 gives each client one condition, α → ∞ gives
every client the same mixture. Without it, results here cannot be compared with
published work, and "non-IID" is a claim with no dial on it.

## What to build

1. **A partitioner registry.** `@partitioner("name")` registers a function that
   turns a request into a list of vehicles. `PARTITIONS` becomes the registry's
   keys, so the CLI choices, the dashboard dropdown and the validation all follow
   from the registration, and adding a strategy touches one function.
2. **`dirichlet`**, with `--alpha`.
3. **A fleet manifest** (`fleet.meta.json`): partition, α, seed, images per
   vehicle. The fleet stage compares the manifest with the config instead of
   inferring intent from labels.

## The variant chosen, and the one rejected

Two things are called Dirichlet partitioning:

- **Per-client mixture (chosen).** Each client i draws p_i ~ Dir(α) over the
  condition groups and fills its budget according to p_i. Shard sizes stay equal, so
  a run's wall clock does not depend on the draw, and `num_examples` — which is
  FedAvg's aggregation weight — stays comparable between vehicles.
- **Per-group split (rejected here).** Each group's images are split across clients
  by one Dir(α) draw. It is equally standard, but it makes shard sizes wildly
  unequal, which conflates two variables — distribution skew and quantity skew — in
  a project whose whole point is watching one thing at a time. Worth adding later as
  `dirichlet-qty` if quantity skew is the subject.

Both are honest; the choice is written down so a future comparison knows which one
produced the numbers.

## Non-negotiable

- Determinism: same seed and α produce the same fleet, and a test asserts it.
- Slices stay disjoint. Overlap would let one image train two vehicles in a round.
- Condition groups come from the existing `PROFILES` predicates — one definition of
  what a driving condition is, not two.
- No change to `my-project/`.

## Verification

- `python -m pytest pipeline/tests -q`
- α = 0.05 concentrates a vehicle on one condition; α = 100 flattens the mixture.
  The test asserts the *direction* of that difference, not a magic number.
- `python -m pipeline.build_fleet --partition dirichlet --alpha 0.3 --vehicles 6
  --per-vehicle 60` prints per-vehicle mixtures, and the shards materialise.
