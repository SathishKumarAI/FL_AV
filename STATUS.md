# federated-yolov8 — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-08-04
- **Where I stopped:** Read the whole repo for the first GPU bring-up (RTX 5070 Ti,
  16 GB). Wrote [`docs/GPU_TESTPLAN.md`](docs/GPU_TESTPLAN.md) — 5 phases, blockers
  B1–B6, gap list. Added `my-project/scripts/populate_images.py` (links BDD100K
  JPEGs into the shards from a pool; self-check passes). No code fixes applied yet.
- **Next action:** Phase 1 of the test plan — conda env + torch **cu128** (not
  cu118; 5070 Ti is Blackwell sm_120), then run the 10-line probe in §Phase 1. It
  confirms or kills B4 (client reads post-training weights off an orphaned module
  ⇒ federation may be learning nothing) and B5 (server `nc=80` vs trainer `nc=13`).
- **Blocked on:** BDD100K images. Repo has all labels + split lists, zero JPEGs.
  Searched 2026-08-04 and ruled out: the `sathishkumar786.ml@gmail.com` Drive
  (no archives at all), the README's Drive link (dead), `origin/images` (empty),
  `origin/laptop_copy` (450-image toy fixture only). Must re-download — Berkeley
  site or Kaggle; the ETH mirror `dl.cv.ethz.ch` did not resolve. Val set alone
  (~1 GB) is enough for the smoke run. Full table in the test plan, Phase 2.
