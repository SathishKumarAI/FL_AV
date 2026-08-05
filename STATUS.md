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

- **Next action:** real BDD100K, then a full-scale Phase 4. Everything is currently
  running on the 10-images-per-shard toy fixture from `origin/laptop_copy`, which is
  enough to prove plumbing and nothing else (mAP50 ≈ 0.05 is noise). Once real images
  land: `python scripts/populate_images.py --pool <dir>`, then raise
  `num_server_rounds`/`local_epochs` and drop `client-resources.num-gpus` below 1.0
  only after checking one client's real VRAM footprint.

- **Environment (this is the part that costs an hour if you forget it):** use the
  **venv at `C:\Users\PRANAS\venvs\fl_yolov8`**, built on python.org 3.12 — *not*
  conda. Smart App Control blocks conda-forge's `_bz2.pyd` on every Python version
  tested; see [`docs/ENV_WINDOWS.md`](docs/ENV_WINDOWS.md). And export
  `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` before `flwr run`, or flwr 1.33
  builds its own runtime env, installs the CPU-only torch wheel, and every client
  silently trains on CPU at 5.5× the wall clock with no error anywhere.

- **Blocked on:** BDD100K images. Repo has all labels + split lists, zero real JPEGs.
  Searched 2026-08-04 and ruled out: the `sathishkumar786.ml@gmail.com` Drive
  (no archives at all), the README's Drive link (dead), `origin/images` (empty).
  `origin/laptop_copy` has the 450-image fixture now in use. Must re-download —
  Berkeley site (account) or Kaggle (`kaggle.json` token); the ETH mirror
  `dl.cv.ethz.ch` did not resolve. Val set alone (~1 GB) is enough. Test plan, Phase 2.
