# Prompt — show the batch the vehicle on the GPU is working through

**Branch:** `feat/live-batch-feed` · **PR:** #34 · **Written:** 2026-08-16
**Stacked on:** `feat/data-consumption-view` (#33) — see
[`2026-08-16-data-consumption-view.md`](2026-08-16-data-consumption-view.md).

## The brief

The Live view narrates a run in numbers: aggregate checksum, mAP by round, GPU
utilisation, pass criteria, a log stream. It has never once shown the data. Ask it
"what is vehicle 3 looking at right now" and there is no answer on the page.

Add exactly one panel that answers it, under three constraints:

1. **Nothing inside `my-project/`.** The client is the ⚠ zone; a live view is not
   worth a gated branch.
2. **No new dependency, no new port, no new process.**
3. **It must be honestly live.** A panel that looks alive while showing a stale
   frame is worse than no panel — that is this project's entire catalogue of
   failures in miniature.

## What made it cheap

Ultralytics rewrites `train_batch{0,1,2}.jpg` at the **start of every `train()`**,
and a round is one `train()` per vehicle. So the file on disk already *is* the batch
currently being consumed — mosaic, scaling and colour jitter applied, which is the
tensor rather than the files. #33 had already built the route that serves it.

The work was therefore: pick the vehicle, point at the file, and stop it going stale.

## The one thing that can go silently wrong

**The browser cache.** Same URL every round means the browser serves round 1's mosaic
for the whole run. The panel updates its heading, the vehicle name changes, and the
picture is a fossil — indistinguishable from working, at a glance, which is the
definition of the failure mode this repo keeps shipping.

So the mtime goes in the URL, and it has its own assertion. Deleting `?t=${f.mtime}`
must fail a test.

## Decisions

| | |
|---|---|
| Refetch the listing every **5 s**, not on every 2 s poll | it globs ten run directories and the picture behind it changes once a round |
| Show only `train_batch*` | `labels.jpg` is a distribution and `val_batch*_pred.jpg` is a prediction. Neither is a batch that was consumed, and putting a prediction under "what it is training on" invites the wrong reading |
| Four states, all explicit | idle, training, started-but-nothing-written-yet, and a vehicle the listing has never heard of. The boring ones are the ones that render an empty box if nobody thinks about them |

## Verification

- `pytest my-project/tests pipeline/tests -q` → 153 passed (154 after the rebase onto
  the merged base).
- A node check against a fake DOM covers all four states, and asserts the mtime is in
  the URL. **Deleting the cache-buster makes it fail**; restored, it passes.
- The two node checks share one runner, `_run_js_check`, which skips when node is
  absent. GitHub's runners ship node, so CI executes both.
- Confirmed against the running server: `index.html` carries `nowTraining` and
  `nowTrainingWho`, `live.js` imports `renderNowTraining`.

## Not verified, and worth saying

- **The panel has never been seen during an actual federation.** Its states are covered
  by the node check; the idle path is what was exercised live. A run with this branch
  checked out is the missing evidence.
- **The rendered page was never looked at.** Browser automation was unavailable for the
  whole session — the extension was not connected and the devtools profile was held by
  a running Chrome. Routes, geometry and empty states are proven by tests; the visual
  is not.
