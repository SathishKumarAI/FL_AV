# Prompt — quantity skew: vehicles that hold different amounts of data

Written before the code. Picks up the deferred half of
[`2026-08-06-dirichlet-partition.md`](2026-08-06-dirichlet-partition.md), which chose the
per-client-mixture Dirichlet variant and wrote down what it left out:

> Worth adding later as `dirichlet-qty` if quantity skew is the subject.

Quantity skew is now the subject. It is *not* being added as a fifth partitioner.

## The problem

Every partitioner hands every vehicle exactly `per_vehicle` images. `condition`,
`random`, `mixed` and `dirichlet` all differ in *what* a vehicle sees and agree
completely on *how much*. Real fleets do not work that way: one car drives twelve hours
a day, another twice a week, and the one parked in a garage contributes nothing that
week at all.

This matters here more than it would elsewhere, because **`num_examples` is FedAvg's
aggregation weight**. Equal shard sizes mean the weights are uniform, which is the one
configuration in which a wrong `num_examples` cannot be noticed. This project has
already shipped that failure twice — a `metrics.csv` reporting 6 308 examples when 10
images existed, and a shard too small for the batch size where no optimizer step ran at
all. Both were invisible partly because every other vehicle reported the same number.

## What to build

**One orthogonal knob, `--size-skew`, not a fifth partitioner.** Skew composes with all
four existing strategies: `condition` *and* uneven sizes is a more realistic fleet than
either alone, and a fifth partitioner could only offer one of them.

1. `Request.size_skew` and `Request.budget(i)` / `Request.val_budget(i)`. Every
   partitioner asks the request how big vehicle *i* is instead of reading
   `req.per_vehicle`. Two call sites: `_by_profile` and `dirichlet`'s `take()`.
2. Sizes drawn lognormally — `exp(gauss(0, skew))` — because real per-vehicle mileage is
   multiplicative, not additive. `skew = 0` is exactly today's behaviour.
3. `--size-skew` on `build_fleet` and `runner`, into `fleet.meta.json`, into
   `ledger.APPROACH_KEYS` and `compare.KEYS` so a skewed run is a distinguishable arm
   rather than a silent repeat of an unskewed one.

## Two decisions that are not obvious

**The fleet total stays `n_vehicles × per_vehicle`.** Sizes are renormalised after the
draw. Skew must vary the *distribution* of the budget, never its size — otherwise a
skewed run and an unskewed one differ in image-visits too, and the comparison is void
for exactly the reason `compare.py` already flags multi-variable configs. This project's
one measured result is a budget-parity claim; skew must not quietly break it.

**Every shard is floored at `max(32, per_vehicle // 10)`.** Below the batch size, no
optimizer step happens and the run is a silent no-op that logs fine — a failure already
in this repo's catalogue. A realism knob that can produce it is a bug generator, so the
floor is not optional and is asserted.

## Non-negotiable

- `size_skew = 0` reproduces today's fleets byte for byte, including not consuming a
  single draw from the rng. A fleet built before this change must not be forced to
  rebuild, and `meta.get("size_skew")` being absent must read as 0, not as a difference.
- Determinism: same seed and skew produce the same sizes.
- Slices stay disjoint.
- The fleet stage's "vehicle below `per_vehicle` images → rebuild" check must learn
  about the floor. Left alone, it would declare every skewed fleet stale and rebuild it
  on every single run — and `build_fleet` rmtree's the shard directory.
- No change to `my-project/`.

## Verification

- `python -m pytest pipeline/tests -q`
- `python -m pipeline.vehicles` self-check.
- Sizes differ, sum to `n × per_vehicle`, none below the floor, slices still disjoint,
  and two calls at the same seed agree.
- `python -m pipeline.build_fleet --vehicles 6 --per-vehicle 60 --size-skew 0.8` prints
  six different `train=` counts that add up.
