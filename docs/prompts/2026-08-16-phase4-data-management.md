# Phase 4 — data management: addressable, clean, cheap to rebuild

**Date:** 2026-08-16 · **Phase:** 4 of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 65 (finish), 66, 69, 70, 71, 74, 44, 75, 76, 36

## Goal

Make a fleet **addressable by content** and **provably clean**, and make rebuilding one
cheap. Today `fleet.meta.json` records partition, α, seed and per-vehicle counts, which
is enough to re-run the generator but not enough to prove two runs saw the same images.
Everything phase 3 concludes rests on that proof.

## Hard constraints

- **Never commit data.** BDD100K is 7.6 GB in the kagglehub cache and shards hardlink to
  it. If this work produces a new kind of artifact — a manifest, a parquet index, a
  quality report — its ignore rule lands in `.gitignore` **in the same commit**, and
  `test_generated_paths_are_all_gitignored` covers it.
- **Never run `build_fleet` while a federation is in flight** — it `rmtree`s
  `pipeline/vehicles/batch/`. The incremental deliverable below is partly there to make
  that trap smaller, not to make it safe.
- **Hash the list, not the pixels.** A SHA-256 over the sorted relative image paths is
  what makes "same fleet" checkable. Hashing 7.6 GB of JPEGs per run buys nothing the
  hardlink target does not already guarantee.
- **The leakage check becomes a stage failure, not a report line.** Fail loudly is rule 5.
- No DVC, no remote cache, no daemon. A content-hash manifest is the part that was
  actually needed and it does not fight the never-commit-data rule.

## Why now — the incident this phase exists to prevent

The fleet on disk was built *before* the holdout existed. `python -m pipeline.validate`
reports that **439 of the 1 000 held-out images sit in vehicles' val splits**. No client
trained on them, so the headline holdout curve is sound — but they did feed clients'
self-evaluation, and that was discovered by a person reading output, not by the pipeline
refusing to run. A gate would have caught it before the GPU hours were spent.

## Deliverables

| # | Item | File |
|---|---|---|
| 1 | **Content-hash fleet manifest** — per shard, the sorted relative image list, its SHA-256, and a fleet-level hash over the shard hashes. Written beside `fleet.meta.json`, quoted in every report and every experiment arm | `pipeline/build_fleet.py`, `pipeline/report.py` |
| 2 | **Leakage gate** — no image in two splits, no image in two shards, holdout disjoint from every shard and from every val split. Promoted from `validate` output to a stage that **halts the chain** | `pipeline/validate.py`, `pipeline/stages.py` |
| 3 | **Incremental populate** — link only what changed; a rebuild with identical parameters is a no-op that preserves `labels.cache` (which phase 1 depends on) | `pipeline/build_fleet.py` |
| 4 | **Parquet attribute index** replacing the 6.7 MB `attributes.json`, read columnar, with the JSON path kept as a fallback for one release | `pipeline/vehicles.py`, `pipeline/paths.py` |
| 5 | **Data-quality audit** — per shard: empty label files, zero-area or out-of-frame boxes, images with no objects, duplicate stems. `num_examples` is FedAvg's aggregation weight and it counts *images*, so a shard of blanks silently buys voting power | `pipeline/dataset_stats.py` |
| 6 | **Per-shard class histogram**, surfaced in the report and the Data tab, plus **per-class mAP** in evaluation. `car` is 55.4 % of all objects and `train` has 29 instances fleet-wide; an averaged mAP is a car detector's report card | `pipeline/dataset_stats.py`, `pipeline/holdout.py` |
| 7 | **Stale-artifact detection** — refuse to score a run whose checkpoint directory contains files older than the run. This has already produced a wrong number once | `pipeline/holdout.py`, `pipeline/verify.py` |
| 8 | **Condition-supply guard** (backlog 67) — refuse a fleet whose per-vehicle size exceeds the rarest condition in the profile, instead of silently topping up with random images and calling the run non-IID | `pipeline/validate.py` |

## The supply table this guard enforces

| Condition | Available |
|---|---|
| night | 31 900 |
| daytime city | 24 878 |
| highway | 19 878 |
| snow | 6 318 |
| rain / fog | 5 951 |
| dawn / dusk | 5 805 |
| overcast residential | **1 419** |
| parking / tunnel | **582** |

1 400 per vehicle is the largest size that keeps all six conditions of the current
profile genuinely distinct. That is a fact the code should assert, not a fact the docs
should remember.

## Definition of done

```bash
python -m pytest pipeline/tests -q                # 59 + new, exit 0
python -m pipeline.build_fleet --vehicles 6 --per-vehicle 1400
python -m pipeline.build_fleet --vehicles 6 --per-vehicle 1400   # second run: no-op, same hash
python -m pipeline.validate                        # exits non-zero on a leaking fleet
python -m pipeline.dataset_stats --audit
git status --porcelain                             # empty: no artifact escaped .gitignore
```

Recorded in the commit body:

- the fleet hash, and proof the second build produced the identical one
- the rebuild wall clock, before and after incremental populate
- the audit's counts per shard: empty labels, zero-area boxes, objectless images
- the leakage gate failing on the known-bad fleet, and passing on a rebuilt one
- attribute-index load time, JSON vs parquet

## Out of scope

A second dataset (backlog 72), perceptual-hash dedup (78), synthetic weather
augmentation (73), active learning (45), label-quality *scoring* as opposed to
*reporting* (79). All of them want the manifest to exist first.
