# federated-yolov8 — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-08-05
- **Where I stopped:** First GPU bring-up **done**. The federation trains on the
  RTX 5070 Ti and its global weights now actually change between rounds — they did
  not before. All of [`docs/GPU_TESTPLAN.md`](docs/GPU_TESTPLAN.md) Phases 1–4 pass;
  results and exact commands are in its §6.

  Fixed: B1–B6 from the test plan, plus three defects only running it exposed —
  **B7** a third `nc=80` model in `_save_global_model` (silently skipped every
  checkpoint), **B8** all Ray actors writing one `logs/client.log` (records vanish,
  so the log lied about which shard trained), and **B9** `configure_fit` mutating the
  single `FitIns` config dict `FedAvg` shares across clients — every client got the
  **last** assigned `batch_id`, so the whole federation trained one shard while the
  server logged two. Guarded by `tests/test_batch_assignment.py`. 23 tests pass.

- **Next action:** full-scale Phase 4 on the real BDD100K. Everything measured so far
  ran on the 10-images-per-shard toy fixture from `origin/laptop_copy`, which proves
  plumbing and nothing else (mAP50 ≈ 0.05 is noise). With real images populated
  (see [`docs/DATASET.md`](docs/DATASET.md)): raise `num_server_rounds`/`local_epochs`,
  and drop `client-resources.num-gpus` below 1.0 only after measuring one client's real
  VRAM footprint. Expect hours — 6 308 images × N clients × epochs is serialised on a
  single card by `num-gpus = 1.0`.

- **Environment (this is the part that costs an hour if you forget it):** use the
  **venv at `C:\Users\PRANAS\venvs\fl_yolov8`**, built on python.org 3.12 — *not*
  conda. Smart App Control blocks conda-forge's `_bz2.pyd` on every Python version
  tested; see [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md). And export
  `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or flwr 1.33
  builds its own runtime env, installs the CPU-only torch wheel, and every client
  silently trains on CPU at 5.5× the wall clock with no error anywhere.

- **Not blocked any more:** the BDD100K images have a working source.
  `kagglehub.dataset_download("solesensei/solesensei_bdd100k")` pulls ~6.5 GB with
  **no Kaggle account or token**, then `scripts/populate_images.py --pool <path>`
  hardlinks them into the shards. Full instructions, plus a table of the sources that
  are dead so nobody searches again, in [`docs/DATASET.md`](docs/DATASET.md).
