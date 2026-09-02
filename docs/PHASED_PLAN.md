# The phased plan — faster runs, better numbers, in that order

Written 2026-08-16, after the run that produced the project's first real result:
**84.5 % of a budget-matched centralised ceiling** (0.4173 vs 0.4936 mAP50 on the
1 000-image holdout), 3 296 s of GPU time, 82.2 Wh.

That run also produced the two numbers this plan exists to attack:

| Measured | Value | What it means |
|---|---|---|
| Mean GPU utilisation | **27 %** | roughly three quarters of the wall clock is not training |
| Peak VRAM | **5 087 MiB of 16 303** | two more clients fit on the card, unused |
| Run-to-run variance (observed) | **≥ ±0.016 mAP50** | larger than several differences the project wants to call results |

So: the experiment loop is both slow and imprecise. Everything below is ordered by
*what unblocks the next thing*, not by size. Speed first, because every later phase is
paid for in runs, and precision second, because until the noise floor is known no
result is a result.

## Order, and why this order

| Phase | Question it answers | Gate to the next phase |
|---|---|---|
| **0 — Measure** | Where does the 73 % of non-training wall clock go? | a per-round timing breakdown exists on disk |
| **1 — Runtime** | How many runs per GPU-hour? | wall clock per image-visit down ≥ 2×, holdout mAP unchanged within noise |
| **2 — Schedule & head** | Why do 24 effective epochs only reach 0.4173? | box/cls/dfl fall *within* a round instead of rising |
| **3 — Evidence** | Is any difference real? | the seed spread is known and printed beside every comparison |
| **4 — Data** | Is the input reproducible and clean? | a fleet is addressable by content hash and reproducible from it |
| **5 — Advanced FL** | Which algorithm suits non-IID driving data? | a leaderboard where the gaps exceed the phase-3 spread |
| **G — GitHub** | Does the repo defend all of the above? | CI is green on Windows *and* Linux, nightly, on `main` |

**Phases 1 and 2 are the ones that change every subsequent number.** Phase 5 is the
interesting one and it is deliberately last: comparing strategies on a slow, noisy,
badly-scheduled trainer measures the trainer.

---

## Phase 0 — measure the round before optimising it

Backlog 95. One run of the demo profile, instrumented, producing a table of where the
seconds go: shard scan, model construction, AMP check, warmup, steady-state training,
validation, checkpoint write, weight serialisation, server aggregation, idle.

**Why it comes first.** The 27 % utilisation figure is a mean over the whole run. It
does not distinguish "the dataloader starves the GPU" from "clients are serialised and
five of them are waiting". Those two have completely different fixes and the plan below
assumes the first — an assumption worth 20 minutes to check.

Deliverable: `pipeline/profile.py` and a stored breakdown per stage. Nothing else in
phase 1 should be believed until this exists.

---

## Phase 1 — runtime: more runs per GPU-hour

Backlog 89, 90, 91, 92. Five levers, all measured against the same fleet fingerprint,
all required to leave holdout mAP unchanged within the phase-3 spread. Ranked by
expected effect on this hardware.

| # | Lever | Where | Why it should work here | Expected |
|---|---|---|---|---|
| 1 | **Pack clients concurrently** — `client-resources.num-gpus = 0.33` | `my-project/pyproject.toml` | peak is 5.1 GB of 16.3 at the 1 400-image profile. Serialisation is a full-scale setting applied at demo scale | ~2–2.5× wall clock |
| 2 | **`cache="ram"`** in the client's `train()` | `client_app.py` ⚠ | 1 400 images at 640 px is ~2–3 GB of RAM. Removes JPEG decode from the step loop — the prime suspect for 27 % utilisation | 1.3–2× per client |
| 3 | **Windows dataloader** — `workers=0` today | `client_app.py` ⚠ | the comment is right that spawned workers deadlock inside a Ray actor, but `workers=0` means decode happens on the training thread. With `cache="ram"` this stops mattering; without it, it *is* the bottleneck | see 2 |
| 4 | **Reuse the label cache across rounds** | `pipeline/build_fleet.py`, shard dirs | Ultralytics writes `labels.cache` beside each shard and rescans when it is missing. 6 vehicles × 6 rounds = 36 scans. The cache survives only if the fleet is not rebuilt between rounds — assert that, do not assume it | seconds × 36 |
| 5 | **Persistent client actors** | Flower client state ⚠ | today the `YOLO` object is constructed from yaml and reloaded every round. Keeping it in node state removes a fixed per-round cost | small, but free |

**The trap this phase must not fall into.** Every one of these can make a run finish
faster *and* train less. Lever 1 changes nothing mathematically — clients are
independent within a round — but levers 2–5 touch the data path. Each is accepted only
if the holdout curve is unchanged and the round-over-round aggregate checksum still
moves. Speed that costs mAP is not speed, it is a shorter run.

**Do not** reach for `torch.compile`, multi-GPU or mixed-resolution training here.
Compile's warmup is paid per process and clients are short-lived; the card is one card;
and mixed resolution changes the result you are trying to hold constant.

Prompt: [`docs/prompts/2026-08-16-phase1-runtime-throughput.md`](prompts/2026-08-16-phase1-runtime-throughput.md).

---

## Phase 2 — the schedule and the head: better mAP per epoch

Backlog 27, 30, 28, 29. ⚠ All of it changes `my-project/`, so: own branch, own prompt.

This is the phase with the largest expected effect on the headline number, and it rests
on three facts checked against the installed ultralytics 8.4.115 rather than assumed:

```
warmup_epochs = 3.0   lr0 = 0.01   lrf = 0.01   cos_lr = False
close_mosaic  = 10    mosaic = 1.0   freeze = None   patience = 100   nbs = 64
```

**Fact 1 — three of every four epochs are warmup.** `local_epochs = 4` against
`warmup_epochs = 3.0` means the LR ramp never finishes. The Metrics tab already shows
box, cls **and** dfl rising across the four epochs of every round: each client ends the
round worse than the aggregate it started from, and FedAvg then averages six models
that each went slightly backwards.

**Fact 2 — the LR schedule restarts every round.** `lrf = 0.01` decays the LR to 1 % of
`lr0` *within* the round, then the next round starts a fresh `YOLO.train()` at `lr0`
again. Six rounds is therefore six independent anneals, not one. The fleet never gets
the low-LR consolidation phase that makes the last epochs of a centralised run count —
which is a decent structural explanation for the 15 % gap to the ceiling, separate from
anything about federation.

The fix is a **server-driven schedule**: the round number is already broadcast, so the
client can set `lr0_round = lr0 · f(round / total_rounds)` and `warmup_epochs ≈ 0.1`
for every round after the first. One global anneal, spread across rounds.

**Fact 3 — the 13-class head starts random.** COCO weights transfer the backbone and
discard the head, so round 1 spends its gradients teaching the head what a car is,
while backpropagating noise into good features. Two independent fixes, cheap, and
testable separately:

- `freeze=10` for round 1 only — protects the backbone while the head settles.
- **Warm-start the head from COCO rows.** BDD100K's 13 classes overlap COCO heavily
  (`person`, `car`, `bus`, `truck`, `train`, `motorcycle`, `bicycle`, `traffic light`,
  `stop sign`). Copying the matching output channels out of the COCO head instead of
  initialising them randomly is a few lines and starts the fleet from a detector rather
  than from noise. This is not in the backlog and should be; it is the highest
  expected-value item in the phase.

Also here, because it is nearly free and saves GPU hours: **early stopping on the
shared holdout** (backlog 29). `patience = 100` inside a 4-epoch round does nothing;
the stopping decision belongs to the server, on the holdout curve, between rounds.

Prompt: [`docs/prompts/2026-08-16-phase2-schedule-and-head.md`](prompts/2026-08-16-phase2-schedule-and-head.md).

---

## Phase 3 — evidence: know the noise floor before claiming a result

Backlog 42, 31, 81. This phase produces no feature. It produces the right to say the
word "better".

The known bad case: a ceiling trained on 14 000 images for 24 epochs (1.667× the
budget) scored **0.4771**, *lower* than the 8 400-image ceiling's **0.4936**. More data
and more compute produced a worse model. Whatever caused that, it puts a floor under
believable differences at roughly ±0.016 mAP50.

| Run | Command | Answers |
|---|---|---|
| Seed repeats | `python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes` | what a difference has to exceed to be real |
| Rounds × epochs at constant product | 12×2, 6×4, 3×8 at identical image-visits | client drift, directly. Phase 2 predicts fewer local epochs win |
| Partition control | `--preset partitions` | how much of the result is federation and how much is just data |

**Report the spread everywhere, not the mean.** The Metrics tab already groups repeats;
after this phase every comparison table in this repo gets a ± column, and any row whose
delta is inside the spread is written as "no measured difference" rather than as a
winner.

Prompt: [`docs/prompts/2026-08-16-phase3-evidence.md`](prompts/2026-08-16-phase3-evidence.md).

---

## Phase 4 — data management: reproducible, clean, cheap to rebuild

Backlog 65 (finish), 66, 69, 70, 71, 74, 44, 75, 36, 76.

The fleet is already deterministic and `fleet.meta.json` records partition, α, seed and
sizes. What is missing is that a fleet is not yet **addressable**: there is no hash you
can quote in a paper that pins the exact image list.

| # | Item | Why it earns its place |
|---|---|---|
| 1 | **Content-hash fleet manifest** — exact relative image list per shard, sorted, SHA-256 over it | makes "same fleet" checkable rather than asserted, and lets phase 3 prove two arms shared data |
| 2 | **Leakage gate** — no image in two splits, no image in two shards, holdout disjoint from every shard, promoted from report to *stage failure* | the last session shipped a run where 439 holdout images sat in vehicles' val splits. It was caught by reading, not by the pipeline |
| 3 | **Incremental populate** — relink only what changed | fleet rebuild is currently rmtree-and-relink, which is why it must never run during a federation |
| 4 | **Parquet attribute index** instead of the 6.7 MB JSON | parsed on every invocation of several tools; columnar read is a fraction of it |
| 5 | **Data-quality audit** — empty label files, zero-area boxes, images with no objects, per shard | backlog 44. These silently reweight FedAvg, since `num_examples` counts images not objects |
| 6 | **Per-class histogram per shard, and per-class mAP** | `car` is 55.4 % of objects, `train` has 29 instances fleet-wide. An averaged mAP is a car detector's report card |
| 7 | **Stale-artifact detection** | already bit the project once: the holdout scorer mixed one run's checkpoints with the previous run's |

Dataset versioning (backlog 69): a content-hash manifest, **not** DVC. DVC adds a
remote, a cache and a daemon to a repo whose hard rule is that data is never committed;
the hash is the part that was actually needed.

Prompt: [`docs/prompts/2026-08-16-phase4-data-management.md`](prompts/2026-08-16-phase4-data-management.md).

---

## Phase 5 — advanced FL, now that a comparison can mean something

Backlog 48, 49, 50, 37, 53, 54, 56, 57, 60, 55.

The strategy registry (backlog 47) already makes twelve Flower strategies reachable by
name. What is missing is not plumbing, it is a fair contest. Ordered by expected value
on *this* dataset:

| # | Technique | Why here | Cost |
|---|---|---|---|
| 1 | **True FedProx** | today's `fedprox` is a weight-space approximation applied after training, which is honest but is not FedProx. See the mechanism note below — it is reachable, contrary to the earlier claim | medium ⚠ |
| 2 | **Personalised heads** (backlog 37) | share the backbone, keep a per-vehicle 13-class head. Condition-partitioned vehicles is the textbook case for this, and it directly attacks the per-vehicle spread (0.4069 → 0.4754) | medium ⚠ |
| 3 | **FedAvgM** | server momentum, one keyword argument, frequently a free win on non-IID | one run |
| 4 | **FedAdam / FedYogi at a real budget** | the existing comparison ran 2 rounds × 1 epoch, where a server-side optimiser has not had time to help. Rerun at 6 × 4 | 2 runs |
| 5 | **Faulty vehicle, then Krum / trimmed-mean** | robustness claims need something to be robust against. Label noise or random weights from one vehicle | medium |
| 6 | **Client sampling and stragglers** | `fraction_fit < 1.0`, vehicles that miss rounds. This is what a real fleet does | small |
| 7 | **Communication cost** — quantise / sparsify, measure the mAP paid | the only axis where FL has an engineering story beyond privacy | medium |
| 8 | **DP wrappers with a stated ε** | the honest privacy story. FL alone is not privacy: gradients leak | medium |

### The mechanism note — true FedProx *is* reachable

`docs/FL_TECHNIQUES.md` said Ultralytics exposes no per-step hook. Checked against
8.4.115, that is half right and the half that is wrong is the useful half:

- `"optimizer_step"` **is** a key in `ultralytics.utils.callbacks.base.default_callbacks`,
  but `BaseTrainer.optimizer_step` never calls `run_callbacks("optimizer_step")`. The
  callback is dead. Registering one buys nothing and looks like it works.
- `BaseTrainer.optimizer_step` **is** an ordinary method, called from `_do_train` after
  `backward()` and before `zero_grad()`, and `YOLO.train()` accepts a `trainer=` class.

So the proximal term goes in a `DetectionTrainer` subclass that adds
`μ · (w − w_global)` to `p.grad` and then calls `super().optimizer_step()` — a real
per-step FedProx, in about fifteen lines, with the global weights captured at round
start.

Prompt: [`docs/prompts/2026-08-16-phase5-advanced-fl.md`](prompts/2026-08-16-phase5-advanced-fl.md).

---

## Phase G — GitHub management, running alongside everything

Backlog 96, 97, 87, 99, 100. Not a phase in sequence; the thing that keeps the other
phases from rotting.

| Item | What it prevents |
|---|---|
| **CI matrix on Windows + Linux** | every defect in this project's history is a path, CWD or encoding trap, and CI runs on one OS |
| **Nightly smoke on `main`** | `main` is green at merge time and never checked again |
| **Branch protection**: PR required, CI required, squash-only, delete branch on merge | the discipline is currently a document, not a rule |
| **Run bundle as a CI artifact** | config + manifest + metrics, so a number in a PR is downloadable |
| **PR body carries the numbers** | the template exists; the required-evidence block does not |
| **One issue per phase, backlog items as a checklist** | `BACKLOG_100.md` is a good list and a bad tracker — it cannot be assigned, closed or linked from a commit |
| **ADRs for choices already made** | assemble-don't-build, the isolation rule, the partition design. Written once, they stop being re-argued |

Prompt: [`docs/prompts/2026-08-16-github-management.md`](prompts/2026-08-16-github-management.md).

---

## What this plan deliberately does not do

- **No architecture change before phase 3.** yolov8s vs 8m vs yolo11s at fixed epochs is
  backlog 34 and it is a phase-6 question. Swapping the model while the schedule is
  broken and the noise floor is unknown produces a comparison that says nothing.
- **No second dataset** until the first one is hash-addressable.
- **No new dashboard surface.** Assemble before building: MLflow owns metrics history
  and is already wired but refused the last run (it needs a SQLite backing store, not
  the filesystem backend). Fixing that is smaller than any panel that would duplicate it.
- **No hosted tracker.** W&B / Comet / Neptune are rejected on the credentials rule,
  not on quality.
