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
  server logged two. Guarded by `tests/test_batch_assignment.py`. 24 tests pass, and CI now runs
  an end-to-end federation smoke job.

- **Next action:** longer full-scale run. The data is in place and a bounded run on
  real BDD100K works: 2 clients x 2 rounds x 1 epoch on full 6 308-image shards took
  919 s and lifted eval mAP50 **0.265 -> 0.311** between rounds. Scale up rounds and
  epochs from there (test plan §6 has the numbers). Keep
  `client-resources.num-gpus = 1.0`: one client peaks at **15.9 GB of 16.3 GB**, so
  clients must stay serialised — 0.5 would OOM.

- **Environment (this is the part that costs an hour if you forget it):** use the
  **venv at `C:\Users\PRANAS\venvs\fl_yolov8`**, built on python.org 3.12 — *not*
  conda. Smart App Control blocks conda-forge's `_bz2.pyd` on every Python version
  tested; see [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md). And export
  `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or flwr 1.33
  builds its own runtime env, installs the CPU-only torch wheel, and every client
  silently trains on CPU at 5.5× the wall clock with no error anywhere.

- **Data: done.** All ten shards hold the real BDD100K (6 308 train + 1 010 val each,
  hardlinked onto the kagglehub cache, zero extra disk).
  `kagglehub.dataset_download("solesensei/solesensei_bdd100k")` pulls 7.6 GB with
  **no Kaggle account or token**, then `scripts/populate_images.py --pool <path>`
  hardlinks them into the shards. Full instructions, plus a table of the sources that
  are dead so nobody searches again, in [`docs/DATASET.md`](docs/DATASET.md).
