# Prompt — a Data tab and a Plan tab

Written before the code. Backlog 75 (per-shard class histogram in the UI), 17
(per-class panel), 21 (progressive disclosure), plus the plan view the run form has
always needed.

## The problem

Two questions the dashboard cannot answer.

**"What is the fleet actually training on?"** The drawer shows one vehicle's mixture
if you click it. Nothing shows the fleet as a dataset: how many images exist, how the
13 classes are distributed, whether `car` dominates every shard (it does), which
conditions are thin, what the holdout contains, or whether the shards are valid. That
information exists — `vehicles.composition`, `pipeline.validate`, the label files —
and none of it is on screen.

**"What is this run going to do, and what will it cost?"** The Control tab has a
one-line estimate derived from a single measured constant. It does not say how many
image-visits the configuration implies, that the centralised ceiling needs the same
number to be comparable, which stages will actually run, or what the equivalent CLI
command is. Somebody planning an experiment has to hold that arithmetic in their head,
and the budget-parity bug that shipped this morning is what happens when they do.

## What to build

### Data tab

| Component | Says |
|---|---|
| Pool readouts | images available, materialised, held out, pooled for the ceiling |
| Class distribution | all 13 classes across the fleet, and per shard when one is selected — `car` dominance is the headline finding here |
| Condition mix | stacked bars per vehicle, from the BDD attributes, so "vehicle 3 is rain / fog" is visible for the whole fleet at once |
| Shard table | sortable: images, labels, classes, fingerprint, condition |
| Holdout card | size, seed, its own class and condition mix — it must look like the data, or the metric is measuring a different distribution |
| Validation | the six checks, green or the exact complaint |
| Sample strip | frames from the selected shard |

### Plan tab

| Component | Says |
|---|---|
| Budget arithmetic | vehicles × images × rounds × epochs = image-visits, and the epochs a matched centralised run needs |
| Stage preview | what will run, what will skip, and why, with the estimated cost of each |
| Cost estimate | wall clock and Wh, from measured constants rather than guesses |
| Commands | the exact CLI for this configuration, and the four comparison presets, each copyable |

## Interactivity

Selecting a shard filters the class chart, the sample strip and the table highlight
together. Table headers sort. Charts keep the existing hover and arrow-key readout.
Commands copy to the clipboard. Everything keyboard reachable, focus visible.

## Non-negotiable

- No build step, no CDN, ES modules, one concern per file: `js/data.js`, `js/plan.js`.
- Reading 14 000 label files must not block the dashboard. Compute once, cache by
  fleet fingerprint, and serve the cache.
- Read-only. The Data tab must not be able to change the data it describes.
- The class histogram is derived from the label files themselves, not from a
  hand-written table that can drift.

## Verification

- `python -m pytest pipeline/tests -q`
- `/api/data` returns real counts for the fleet on disk, and the second call is served
  from cache.
- Both tabs rendered in a browser at 1280px with a real fleet, screenshotted, no
  console errors.
