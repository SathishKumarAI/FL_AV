# Build prompt — per-vehicle learning visualisation, plus fixes

**Design:** [`2026-08-05-pipeline-observability-design.md`](../superpowers/specs/2026-08-05-pipeline-observability-design.md) (revision 2)
**Branch:** `feat/pipeline-observability`
**Follows:** [`2026-08-05-pipeline-observability-build-prompt.md`](2026-08-05-pipeline-observability-build-prompt.md)

## Goal

Make it visible **how each individual vehicle learns from its own slice of the
world** — not just that the fleet's aggregate moved. Today the dashboard answers "did
the federation learn?"; it should also answer "did *this* vehicle learn, from *this*
kind of data, and how did that differ from the others?"

That question is the entire reason the fleet is condition-biased. Right now the
divergence between a night-driving vehicle and a highway one is real in the data and
invisible in the UI.

## The data already exists — do not add logging to my-project

Verified present after the 2026-08-05 six-vehicle run:

| Source | Grain | Fields |
|---|---|---|
| `logs/client.*.log` → `[Client] <bid> Training done. metrics={...}` | one per vehicle **per round** | precision, recall, mAP50, mAP50-95, fitness |
| `runs/detect/runs/fl/batch<N>/results.csv` | one per vehicle **per epoch** | `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `val/*_loss`, `lr/pg0` |
| `logs/client.*.log` → received/sent checksums | one pair per vehicle per round | how far that vehicle moved the weights |
| `pipeline/vehicles/fleet.json` | static | condition, shard size |

Note the doubled path (`runs/detect/runs/fl/...`) — that is Ultralytics prepending its
settings dir to the `project=` we pass. Do not "fix" it by editing `client_app.py`.

Note also that `results.csv` is **overwritten each round** (same `name=`, `exist_ok=True`),
so it holds the most recent round's epochs. Cross-round history must come from the
client-log line, not from the CSV. Say so in the UI rather than implying otherwise.

## Deliverables

### 1. `pipeline/vehicle_metrics.py` (new)

- `per_vehicle_rounds()` → `{vid: [{round, precision, recall, mAP50, mAP50_95, fitness}]}`
  parsed from the `Training done. metrics=` lines. Parse the dict **safely**
  (`ast.literal_eval`, never `eval`).
- `per_vehicle_epochs()` → `{vid: [{epoch, box_loss, cls_loss, dfl_loss, mAP50, ...}]}`
  from each vehicle's `results.csv`.
- `weight_movement()` → `{vid: [|sent - received| per round]}` — how much each vehicle
  actually changed the model it was handed.
- `divergence()` → per-vehicle mAP50 minus the fleet mean, per round. This is the
  number that makes non-IID data legible.

### 2. UI components (Live view)

- **Vehicle learning small multiples** — one mini chart per vehicle showing its mAP50
  across rounds, drawn on a shared y-axis so they are visually comparable. Label each
  with its condition.
- **Fleet comparison** — all vehicles' mAP50 overlaid on one chart, so divergence is
  a shape you can see rather than numbers to compare.
- **Divergence bars** — signed bars, per vehicle, above/below the fleet mean.
- **Contribution** — each vehicle's share of total `num_examples`, which is literally
  its FedAvg weight.
- **Loss curves** — box/cls/dfl for the selected vehicle, from `results.csv`.
- Clicking a vehicle card selects it and focuses the loss curves on it.

Charts stay inline SVG, no CDN, no chart library. Keep the existing visual language.

### 3. Report

A **per-vehicle section**: condition, shard size, mAP50 per round, final mAP50,
divergence from fleet mean, weight movement, and contribution share. Both HTML and
Markdown, from the same dict.

### 4. Fixes carried from the last run

1. `fleet.json` records `n_train` but not `n_val`, so the report prints `val | ?` for
   every vehicle. Write both.
2. **Report generation failures are invisible.** `_write_report` catches the exception
   and emits a `log` event, which a filtered console view drops — so the last run
   produced no report and said nothing about it. Make it a first-class, loud failure:
   surface it in the run result, and never let the only trace be a filtered log line.
3. The telemetry dict passed into the report omits `peak_power_w` and
   `mean_util_pct`, which then render as `None`. Pass the whole summary.

## Hard constraints

Unchanged from the previous prompt, and they still bind:

1. **Do not modify `my-project/`.** Read its outputs; run its scripts. Enforced by test.
2. **Do not add logging to `my-project`** to make this easier. Every field above
   already exists.
3. **No new heavyweight dependency.** No chart library, no dataframe library — `csv`
   and `ast` from the stdlib are enough.
4. **No data, no credentials committed.** New artifact kinds get ignore rules in the
   same change.
5. **Fail loudly** — which is precisely what fix #2 is about.

## Definition of done

```bash
python -m pytest pipeline/tests -q          # all pass, including new per-vehicle tests
python -m pipeline.report                   # per-vehicle section populated, no "?" or None
python -m pipeline.runner --list            # unchanged behaviour
```

Plus: a fresh demo run whose runner-generated report contains the per-vehicle section,
and a dashboard showing six distinguishable learning curves.

Tests must cover: the metrics dict is parsed without `eval`; a vehicle that never
trained does not crash the aggregation; divergence sums to ~0 across the fleet; and
missing `results.csv` degrades to "no epoch data" rather than an exception.

## Out of scope

Per-class metrics, per-image inspection, comparing runs against each other, and any
change to how training itself works.
