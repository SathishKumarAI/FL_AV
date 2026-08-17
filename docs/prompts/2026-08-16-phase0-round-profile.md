# Phase 0 — measure the round before optimising it

**Date:** 2026-08-16 · **Phase:** 0 of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 95

## Goal

Answer one question with a number rather than a guess: **where does the 73 % of
non-training wall clock go?**

The reference run held the GPU at 27 % mean utilisation. That mean is compatible with
two completely different worlds, and they have opposite fixes:

| World | Signature | Fix |
|---|---|---|
| Clients are serialised and five of six wait | client-busy time ≈ wall clock, but only one client busy at a time | `num-gpus = 0.33` (phase 1, lever 1) |
| The dataloader starves the GPU inside `train()` | almost all wall clock *is* inside `train()`, and the card still idles | `cache="ram"`, workers (phase 1, levers 2–3) |

Phase 1's plan assumes the second. That assumption is worth twenty minutes to check,
and phase 1 must not be believed until it is checked.

## Hard constraints

- **`pipeline/` must not modify `my-project/`.** The profiler reads log lines that are
  *already emitted*, exactly as `logparse.py` does. No new logging on the other side of
  the boundary, no instrumentation commit smuggled in.
- **No new dependency.** `datetime.strptime` over lines this repo already writes.
- **A run this project has already done must be profilable.** The deliverable is
  useless if it needs a fresh instrumented run to say anything — the 3 296 s reference
  run is on disk and it is the run whose 27 % started this.
- **No estimate presented as a measurement.** Anything the timestamps do not support
  (per-epoch warmup vs steady-state, JPEG decode as a share of the step) is reported as
  unaccounted, not modelled.

## Inputs — the markers already in the logs

Verified against `my-project/logs/client.45544.log` and `server.30716.log` from the
six-round run of 2026-08-06:

```
00:57:35,355 [Client] Creating FlowerClient instance from client_fn.
00:57:35,513 [Client] YOLO model loaded successfully.
00:57:35,530 [Client] Received weights with checksum: 705.23
00:57:35,539 [Client] Successfully applied received weights to model
00:57:35,545 [Client] Starting local training with batch_id=8, local_epochs=4
00:58:55,185 [Client] 8 Training done. metrics={...}
00:58:55,190 [Client] Sending back weights with checksum: 236.48
...
01:05:16,263 [Server] Aggregating 6 fit results and 0 failures
01:05:16,408 [Server] Aggregated parameters with checksum: 159.89
01:05:16,671 [Server] Saved global checkpoint: checkpoints\global_round_1.pt
```

Every phase boundary phase 0 needs is a pair of those, timestamped to the millisecond.

## Deliverables

| # | File | What it does |
|---|---|---|
| 1 | `pipeline/profile.py` | pairs the markers into intervals, sums seconds per phase, prints a table with each phase's share of the wall clock, and writes `pipeline/.state/profile-<stamp>.json` |
| 2 | the verdict line | the profiler itself names which of the two worlds the run was in, from `train_share` and `max_concurrent_clients` — not left to the reader |
| 3 | `pipeline/tests/test_profile.py` | a test named for the failure it catches, against captured real lines |

## The two numbers that decide phase 1

- **`train_share`** — seconds inside `Starting local training … Training done`, over
  wall clock. High means the overhead is *inside* training and levers 2–3 are the ones
  that matter.
- **`max_concurrent_clients`** — the most client episodes overlapping at any instant.
  `1` proves serialisation and makes lever 1 the whole result, whatever `train_share`
  says.

Both are reported; neither is inferred from the other.

## Definition of done

```bash
python -m pytest pipeline/tests -q          # exit 0
python -m pipeline.profile                  # prints the breakdown of the last run
python -m pipeline.profile --json           # writes .state/profile-<stamp>.json
```

The commit body carries the table for the 2026-08-06 reference run, and the verdict.

## Out of scope

Changing anything the profiler measures. Phase 0 ships a measurement and no lever; a
commit that does both is a commit whose lever cannot be attributed.
