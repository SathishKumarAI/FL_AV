# Engineering Notes — FL_AV Production Hardening

Working design log and decision record for taking FL_AV (federated YOLOv8 on
BDD100K, via Flower) from a research prototype to a correct, deployable system.
Written from an ML + ML-infra engineering standpoint: what was wrong, why it
mattered, what we changed, and what's next. Lead with the decision; the "why" is
in the tables.

Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [RUNNING.md](RUNNING.md) ·
[DEPLOYMENT.md](DEPLOYMENT.md) · [FEDPROX.md](FEDPROX.md).

---

## 1. The headline problem: the "federated" model wasn't actually federating

**Symptom framing:** the system trained and aggregated, but the global model
behaved as if clients barely shared learning.

**Root cause (ML bug):** weight transport used `model.parameters()`, which
serializes **only learnable tensors**. YOLOv8 (Ultralytics `DetectionModel`) is
BatchNorm-heavy, and BatchNorm carries non-learnable **buffers** —
`running_mean`, `running_var`, `num_batches_tracked` — that are *not* in
`parameters()`. So FedAvg averaged the conv/linear weights but **never the BN
running statistics**. At inference each client used its own local BN stats over
a globally-averaged body: silent train/serve skew, poor generalization, results
that look "almost working" but never converge like real FL.

**Fix:** serialize the full `state_dict()` (params **and** buffers) as an ordered
ndarray list; reconstruct on the receiver by zipping against locally-recomputed
`state_dict().keys()` and `load_state_dict(strict=True)`. Integer buffers
(`num_batches_tracked`, int64) are cast to the model's own dtype so the strict
load passes. (PR #21.)

**Why ordered-list + local keys, not a key→array dict over the wire:** Flower
transports an ordered list of ndarrays. Both sides build the *identical*
architecture (`YOLO("yolov8s.pt").model`, `nc=13`) before loading, so positional
order is deterministic and stable. `strict=True` turns any future drift into a
loud failure instead of a silent partial load.

| Decision | Why |
|----------|-----|
| `state_dict()` over `parameters()` | Buffers (BN stats) must be aggregated for correct FL |
| Positional list, keys rebuilt locally | Matches Flower's transport; no key serialization needed |
| `strict=True` load | Fail loud on architecture drift, never silently partial-load |
| Cast int buffers to model dtype | `load_state_dict` dtype check would otherwise reject int64→float |

---

## 2. Aggregation weighting was a constant

FedAvg weights each client's update by `num_examples`. The client hardcoded
`num_examples = 10` for everyone, so aggregation was a plain (unweighted) mean.

- **Impact today:** low — the shards are near-equal (~6308 train images each), so
  the bug barely changed results. **This is why it went unnoticed.**
- **Impact in production:** real cross-silo FL has heterogeneous shard sizes; a
  constant weight biases the global model toward small-data clients.

**Fix:** `count_shard_examples(batch_id, split)` counts the split list file
(`train.txt`/`val.txt`), floored at 1 so no client gets a zero weight; `fit`
reports train count, `evaluate` reports val count. (PR #22.)

---

## 3. Non-IID data → FedProx option

BDD100K shards differ by scene/time-of-day/weather → **non-IID**, where vanilla
FedAvg can diverge as local models drift. We added an optional FedProx-style
proximal term, config-selectable (`strategy=fedprox`, `proximal_mu`).

**Honest limitation:** true FedProx adds `(mu/2)||w−w_global||²` to the loss at
**every SGD step**. Ultralytics' high-level `YOLO.train()` exposes no per-step
loss hook, so we apply the proximal pull **at round level in weight space**
(`w ← w − mu·(w − w_global)`). It's a documented approximation, not exact
FedProx. Exact FedProx needs a custom Ultralytics trainer — tracked below.
(PR #20, [FEDPROX.md](FEDPROX.md).)

---

## 4. Persistence: the trained model was being thrown away

The server aggregated in memory and never saved — after the final round the
federated model was gone, and there was nothing to deploy.

**Fix:** the strategy saves the aggregated model in `aggregate_fit` on a
configurable cadence and always on the final round → `checkpoints/global_round_N.pt`
plus a stable `global_last.pt`. Loading uses the (now buffer-complete)
`set_weights`, so checkpoints carry correct BN stats. Save errors are logged,
never fatal (a checkpoint failure must not abort a federation). (PR #26.)

**Verified empirically:** perturb a BN `running_mean` → `set_weights` →
`YOLO.save` → reload → the buffer survived and the file reloads as a usable
model.

---

## 5. Flower version: 1.13.1 → 1.31.0

The project was pinned **18 minor releases behind** (1.13.1 vs 1.31.0).

**Key API context:** current Flower (1.20+) favors a **Message API** — `ServerApp()`
+ `@app.main()` + `strategy.start(grid=..., initial_arrays=ArrayRecord(...))`,
with `ArrayRecord`/`ConfigRecord`. FL_AV uses the **legacy** API:
`ServerApp(server_fn=...)` + `ServerAppComponents` + a `FedAvg` subclass
overriding `configure_fit`/`aggregate_fit`.

**Decision: bump now, migrate later.** Empirically the legacy API **still imports
and constructs under 1.31.0** (verified: both `ServerApp` and `ClientApp` build;
all `flwr.common`/`flwr.server` symbols resolve), so the upgrade is a low-risk
version bump — no rewrite required. Also dropped `flwr-datasets` (unused; it
pinned an old version blocking the bump). (PR #29.)

| Option | Verdict |
|--------|---------|
| Stay on 1.13.1 | Rejected — far behind, missing fixes/security |
| Bump to 1.31, keep legacy API | **Chosen** — verified working, minimal risk |
| Rewrite to Message API now | Deferred — large change; do it deliberately, with runtime tests |

**Migration sketch (future):** `CustomBatchStrategy` → a `@app.main()` that calls
`FedAvg(...).start(grid, initial_arrays=ArrayRecord(model.state_dict()), ...)`;
clients become `@app.train()/@app.evaluate()` handlers receiving/returning
`ArrayRecord`. Our state_dict-based weight handling maps cleanly onto
`ArrayRecord.to_torch_state_dict()` / `ArrayRecord(state_dict)` — so the BN-buffer
fix is forward-compatible and actually *simplifies* under the new API.

---

## 6. Infra: simulation-only → deployable

| Concern | Before | Now |
|---------|--------|-----|
| Execution | in-process sim only (`local-simulation`) | + `remote-deployment` federation (SuperLink + N SuperNodes, gRPC) |
| Transport security | none | TLS (`gen_certs.sh`, SAN-aware), root cert in federation |
| Packaging | none | `Dockerfile.superlink` (CPU), `Dockerfile.supernode` (CUDA, asserts GPU) + compose |
| Data locality | n/a | each SuperNode mounts only its own `batch_N` shard |
| Portability | hardcoded Windows path; mutated tracked `data.yaml` | `FL_AV_DATA_ROOT` env; `materialize_data_yaml` writes gitignored runtime copy |
| Logging | unbounded `FileHandler` | `RotatingFileHandler` (10MB×5), env-tunable |
| Observability | text logs only | per-round `metrics.csv` + weighted aggregation + summary |
| Persistence | none | global-model checkpoints |
| Tests/CI | none | pytest (17) + GitHub Actions on every push/PR |

**Not yet runtime-verified** (no SuperLink host / certs / GPU / object store in
this environment): the deployment artifacts are syntax/compile-checked and the
correctness logic is unit-tested; the distributed bring-up must be validated on
real infra. This is stated wherever it applies — no silent "it works" claims.

---

## 7. Data & model management (Phase D, planned)

**Problem:** ~326 MB of `batch/` images and ~75 MB of `.pt` weights are committed
to git (plus duplicates). Anti-pattern: bloats every clone, no versioning, churns
history.

**Plan:** DVC with an S3-compatible remote — `dvc add batch/` and the `.pt`
files, commit the small `.dvc` pointers, `dvc pull` at runtime; gitignore the
blobs; publish trained `checkpoints/` to a model registry/object store.
Duplicate weights move to `archive/` (per the move-don't-delete policy), not
deleted. **History purge** (`git filter-repo`) is destructive + force-push →
gated on explicit human go-ahead, done once, all clones re-pulled.

---

## 8. Testing strategy

Unit tests target the **correctness invariants**, designed to run without the
heavy stack (torch-only; ultralytics stubbed) so CI is fast:

| Test file | Guards |
|-----------|--------|
| `test_weights.py` | BN buffers round-trip; `state_dict` count > param count; length-mismatch fails safe |
| `test_num_examples.py` | real shard counts (6308/1010), ≠ constant, ≥1 floor |
| `test_paths.py` | env override; runtime yaml written without mutating tracked file; no Windows path |
| `test_checkpoint.py` | save cadence (every-N + always-final) |
| `test_logging.py` | rotating+bounded handler; no dup handlers; no propagation |

Plus manual integration checks needing the full stack (run with a CPU venv):
real YOLO checkpoint save/reload preserving BN stats. **Gap:** no end-to-end
simulation test in CI yet (needs ray + a tiny fixture dataset) — see roadmap.

---

## 9. Risks & open questions

| Risk | Mitigation / status |
|------|---------------------|
| state_dict key order drift between server/client | identical arch construction + `strict=True` (loud fail) |
| Ultralytics resolves `data.yaml` `path: .` differently than expected | `materialize_data_yaml` writes an **absolute** path → resolution-independent |
| GPU/CUDA/torch/torchvision ABI mismatch in containers | pin the triad to the host driver; supernode asserts `torch.cuda.is_available()` at start |
| TLS SAN ≠ dialed address → handshake fails | `gen_certs.sh` sets SAN; documented; rotation needs SuperLink restart |
| Legacy Flower API removed in a future 2.x | pin `<2.0.0`; Message-API migration sketched in §5 |
| FedProx is a round-level approximation | documented; exact version needs a custom trainer |
| No e2e FL test in CI | add a 1-round CPU sim smoke test with a tiny fixture |

---

## 10. Roadmap (prioritized)

1. **e2e simulation smoke test in CI** — 1 round, 2 supernodes, tiny fixture
   dataset; assert weight checksums move and a checkpoint is written. Highest
   confidence-per-effort next step.
2. **Phase D data/model hygiene** — DVC + registry; history purge (gated).
3. **Exact FedProx** — custom Ultralytics trainer adding the proximal term to the
   per-step loss.
4. **Message-API migration** — adopt `@app.main()`/`strategy.start()`/`ArrayRecord`
   (state_dict handling already compatible).
5. **Secure aggregation / DP** — Flower SecAgg+ and/or DP-SGD for stronger
   privacy guarantees beyond "raw images stay local".
6. **Real `num_examples` from the dataloader** — count actually-trained images per
   round rather than shard-file lines (handles augmentation/sampling).
7. **Observability** — ship metrics to Prometheus/W&B; structured (JSON) logs.

---

## Appendix: ADR index

- **ADR-1** Full `state_dict` weight transport (params+buffers). Accepted. §1.
- **ADR-2** Real `num_examples` from shard size. Accepted. §2.
- **ADR-3** Round-level FedProx approximation under Ultralytics. Accepted (with
  documented limitation). §3.
- **ADR-4** Server-side global checkpointing in `aggregate_fit`. Accepted. §4.
- **ADR-5** Bump Flower to 1.31 keeping the legacy API; defer Message-API
  migration. Accepted. §5.
- **ADR-6** Non-mutating `materialize_data_yaml` + `FL_AV_DATA_ROOT`. Accepted. §6.
- **ADR-7** Archive (don't delete) retired code under `archive/`. Accepted.
