# Flower × YOLOv8: what this pairing actually needs

Written 2026-09-02. Every claim here was checked against the installed
**ultralytics 8.4.115 / torch 2.11.0+cu128** in the project venv, on the RTX 5070 Ti
(sm_120, 16 303 MiB), by running it. Where a number is quoted it was measured on
`batch_1` — 1 400 train / 280 val images, imgsz 640, batch 16, one epoch.

The existing plan treats "federated YOLO" as a plumbing problem that is already
solved, with the remaining work being which Flower strategy to pick. That is the wrong
frame. Three things about *detection specifically* are mis-set today, and none of them
are strategy choices.

---

## 1. `optimizer="auto"` discards `lr0`. Every LR conclusion in this repo names a
## learning rate that was never used

The client calls `yolo.train()` without an `optimizer` argument, so it gets
Ultralytics' default `optimizer="auto"`. Verbatim from the run log:

```
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and
           determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000588, momentum=0.9) with parameter groups
           57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
```

So the actual optimiser is **AdamW at lr 0.000588**, not SGD at 0.01. This invalidates
the framing of two things the project currently believes:

| Believed | Actually |
|---|---|
| "Round 1 costs the warm-started model 0.066 mAP50 **at `lr0 = 0.01`**" | the damage is real and measured, but it happened at AdamW 5.88e-4. The number named as the cause was not in effect |
| "Try a lower `lr0` for a warm-started head" | **setting `lr0` alone does nothing.** It is ignored and logged as ignored, in a line nobody read. A classic silent no-op of exactly the kind this repo collects |

### What `auto` actually substitutes, and what it leaves alone

```python
# build_optimizer, when name == "auto"
lr_fit = round(0.002 * 5 / (4 + nc), 6)                       # nc=13  ->  0.000588
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
self.args.warmup_bias_lr = 0.0
```

| knob | under `auto` |
|---|---|
| `lr0` | **replaced** by `lr_fit`, which depends only on `nc`. Whatever you passed is discarded |
| `momentum` | replaced |
| `warmup_bias_lr` | forced to 0.0 |
| `optimizer` | AdamW or MuSGD, by iteration count |
| **`lrf`** | **untouched.** `_setup_scheduler` reads `args.lrf` directly, so this one still applies |
| `warmup_epochs` | untouched |

**So PR #53's negative result is not the experiment it reports.** That branch computes
`lr0_round` and `lrf_round` per round and passes both — but never sets `optimizer`, so
`lr0_round` was discarded on every round while `lrf_round` applied. Its
"−0.0079 mAP50, negative at six of six rounds" measured varying the *final-LR ratio*
with the starting LR pinned at 0.000588, which is not "one anneal across the run". The
branch is marked do-not-merge and should stay that way, but its **result should be
struck, not carried forward** — it is currently the reason the project believes a global
anneal does not help.

### The threshold to keep an eye on

`iterations = ceil(len(dataset) / max(batch, nbs)) * epochs`, and `nbs` is 64, so the
divisor is 64 rather than the batch size. At the profile everything has been measured
at, both arms sit well under 10 000 and therefore get the *same* AdamW at the *same*
5.88e-4:

| arm | iterations | picked |
|---|---|---|
| client, 1 400 images × 4 epochs | 88 | AdamW 5.88e-4 |
| centralised ceiling, 8 400 pooled × 24 epochs | 3 168 | AdamW 5.88e-4 |

That is worth stating explicitly, because it was worth checking: the headline
"84.5 % retained" is **not** contaminated by an optimiser mismatch.

It would be at full scale. A centralised ceiling over 6 × 6 308 = 37 848 pooled images
for 24 epochs is 14 208 iterations, over the threshold, so it would silently train with
**MuSGD at lr0 = 0.01** while the clients stayed on AdamW at 5.88e-4 — a 17× LR gap and
a different optimiser, chosen by a magic number in a library, in the one run this
project exists to produce. Nothing warns.

**Fix, and it is a precondition for all of phase 2 and for the full-scale run:** pass
`optimizer` explicitly on both sides. Only then does a server-driven `lr0` mean
anything, and only then is the ceiling comparable to the federation by construction
rather than by coincidence.

## 2. FedAvg is averaging BatchNorm running statistics across weather conditions

`get_weights` serialises the full `state_dict` — deliberately, and the docstring
explains why: buffers include BN `running_mean` / `running_var`, and dropping them
would make the federated model wrong. That reasoning is correct for IID clients.

This fleet is **not** IID, and specifically it is partitioned by *condition* — night,
rain, overcast, clear. That is **feature-shift** non-IID, not label-shift: the class
distribution is broadly similar, but the input statistics differ per vehicle, which is
exactly what BN running stats encode. Averaging a night vehicle's `running_mean` with a
clear-daylight vehicle's produces normalisation statistics that describe no vehicle's
data.

This is the case **FedBN** was written for: share every weight *except* the BatchNorm
layers, keep those local and per-vehicle. It costs nothing to run — it is a filter on
which tensors travel — and it targets this project's non-IID axis directly, unlike
FedProx/FedAdam/FedAvgM which all address client drift in *parameter* space.

It is not in `docs/PHASED_PLAN.md`'s phase-5 table. On this dataset it is the highest
expected-value entry in it, and the cheapest.

### Measured, 2026-09-02 — on an IID fleet, which is the wrong fleet for it

Implemented (`local_bn` run-config flag; `batchnorm_keys` + `set_weights(keep_local=)`)
and run against the **random** partition, because that is the fleet the project moved to.
6 vehicles × 1 400 images, 2 rounds × 1 epoch, same holdout:

| round | FedAvg | FedBN | Δ mAP50 |
|---|---|---|---|
| 1 | 0.1045 | 0.1085 | +0.0040 |
| 2 | 0.1201 | 0.1218 | +0.0017 |

**No measured difference.** Both deltas sit far inside the ±0.016 spread, and this is
the predicted result, not a disappointment: random partitioning makes every client's
input distribution the same, so there is no feature shift for a local BatchNorm to
preserve. FedBN was given nothing to do.

The comparison is weak for a second reason worth stating. Under FedBN the saved
checkpoint still carries the *averaged* BatchNorm — no vehicle's — so a holdout score on
it measures a model that never existed. The FedBN column above is therefore a lower
bound on the method even where the method applies.

**The run that would actually test it** is the same pair on the **condition** fleet
(`--partition condition`, fingerprint `7170c3ee9350`), reported per vehicle rather than
on the averaged-BN checkpoint, and only once the phase-3 seed spread is known. Until
then the honest statement is: implemented, wired, tested, and **unmeasured on the
partition it is for**.

**Note the interaction with the holdout.** With BN kept local there is no single global
model to score, so the phase-5 leaderboard must state which of two things it reports:
the shared backbone plus each vehicle's own BN (personalised, scored per vehicle and
averaged), or the shared backbone plus BN re-estimated on the holdout. They are
different claims. This is the same trap `docs/PHASED_PLAN.md` already flags for
personalised heads, and the two techniques stack.

## 3. The round pays for validation three times

Per client per round, at `local_epochs = 1`, measured against the trainer source:

| Pass | Where | Escapable? | Worth escaping? |
|---|---|---|---|
| training epoch | `_do_train` | no — it is the work | — |
| validation | `_do_train`: `if self.args.val or final_epoch or ...` | **no.** `final_epoch` forces it, so `val=False` does not skip it at 1 epoch | — |
| validation again | `final_eval()`, unconditional after the loop, on `best.pt` | yes — it no-ops when `self.best` does not exist, i.e. `save=False` | **no.** Measured at 22.8 s vs 22.9 s. Not where the time goes |
| validation a third time | Flower's `evaluate()` → `yolo.val()`, same val split | yes — `fraction_evaluate < 1.0` | **yes** — phase 0 measured it at 13.8 % of wall clock |

`fraction_evaluate` is **never set** in `server_fn`, so it defaults to 1.0: every
client re-scores itself every round. Phase 0 measured that at **13.8 % of wall clock**,
spent on the metric this project already calls the flattering one. The holdout is what
gets reported. This is the cheapest speed lever in the repo and it is one run-config
key.

### The pass that is not in that table, and cost the most

`plots=True` is the Ultralytics default, and the client never overrode it. Every
`train()` call draws `labels.jpg`, `train_batch{0,1,2}.jpg`, and at `final_eval` the
confusion matrix and the P/R/F1/PR curves — **1.19× of the round**, per client, per
round.

They were being drawn six times to be kept once. The client passes `exist_ok=True`, so
every round writes into the same `runs/fl/batch{n}` directory the next round
overwrites, and `pipeline/train_artifacts.py` — the only thing that reads them — says
so in its own docstring: *"a vehicle's directory holds only its **last** round. These
are not a history."* Rounds 1..n−1 paid GPU time for files destroyed before anything
read them.

Fixed by having the server send `plots` and set it True on the final round only. The
dashboard gets byte-for-byte what it got before.

## 4. `workers=0` is right, and the recorded reason is wrong

The client's comment says spawned dataloader workers deadlock inside a Ray actor. Two
measurements say otherwise, and the correct reason is more useful.

**Measured dataloader cost, dataset `__getitem__` only, no model, no GPU:**

| configuration | ms/sample |
|---|---|
| stock (`mosaic=1.0`, `erasing=0.4`) | **7.93** |
| `mosaic=0.0` | 5.56 |
| `mosaic=0.0 erasing=0.0 scale=0.2` | 5.57 |

At batch 16 that is ~127 ms of CPU per batch, on the training thread, against a GPU
step of the same order. **That is the missing explanation for 27–31 % utilisation** —
neither of the two suspects in `PHASED_PLAN.md` phase 1 covered it, and the plan says
so.

It also explains why `cache="ram"` measured *slower*: caching removes JPEG decode but
leaves mosaic assembly and `random_perspective` — the warps, not the reads. Mosaic
composites four images onto a doubled canvas per sample, and `close_mosaic=10` fires
only at `epoch == epochs - close_mosaic`, which for a 1–4 epoch round is negative.
**Mosaic never closes inside a federated round.**

**But moving that work to workers does not pay, on Windows, at this round length.**
One epoch on `batch_1`, one run per arm:

| arm | wall | util | vs baseline |
|---|---|---|---|
| `workers=0` (the client today) | 36.2 s | 20.4 % | 1.00× |
| `workers=4` | 34.3 s | 20.4 % | 1.06× |
| `workers=8` | 40.1 s | 27.4 % | **0.90×** |
| `workers=4 mosaic=0` | 30.0 s | 29.9 % | 1.21× |
| `workers=4 plots=False save=False` | 25.6 s | 37.0 % | 1.41× |
| **`workers=0 mosaic=0`** | **24.9 s** | 27.2 % | **1.45×** |

Read the last two rows together. Turning mosaic off is worth **more with no workers
(1.45×) than with four (1.21×)** — the workers are a net cost that the mosaic saving
has to pay for first. Windows has no `fork`: every worker spawns a fresh interpreter
and re-imports torch and ultralytics, and a federated round is one to four epochs, far
too short to amortise that. `workers=8` is slower than doing the work inline while
showing *higher* utilisation, which is the cleanest possible demonstration that
utilisation is not the objective — wall clock is.

The utilisation column is worth reading the same way. `workers=0 mosaic=0` is the
fastest arm and only reaches 27.2 %, while a slower arm reaches 37.0 %. Chasing the
27 % number, which is what phase 1 set out to do, would have ranked these backwards.

So: keep `workers=0`, and correct the comment — the reason is spawn cost on short
rounds, not a Ray deadlock, and the deadlock claim did not reproduce. The thing to
attack is the **7.93 ms itself**, by turning augmentation down.

**Those are one run per arm, and one run per arm was not enough.** Repeating the
candidate arms three times each, interleaved, changed the answer:

| arm | median | spread | util | vs baseline |
|---|---|---|---|---|
| baseline (the client today) | 27.2 s | **7.6** | 25.4 % | 1.00× |
| `plots=False` | 22.9 s | 0.3 | 31.2 % | **1.19×** |
| `plots=False save=False` | 22.8 s | 0.4 | 29.9 % | 1.20× |
| `plots=False save=False mosaic=0` | 20.4 s | 0.9 | 32.4 % | **1.34×** |

The single-run pass suggested `plots=False` was worth 1.52×. It is worth **1.19×**.
The difference was one cold start: the baseline's three repeats were 34.6 / 27.1 /
27.2 s, and the 34.6 is the first `train()` of the process — CUDA context, cuDNN
autotune, the AMP check. Because arms run in a fixed order within each repeat, the
first arm of the first repeat always eats that, which is a flaw in the harness and not
a property of the arm. Warm baseline is 27.1–27.2 against `plots=False` at 22.7–22.9:
non-overlapping, and the 7.6 s "spread" is a cold-start artifact rather than run-to-run
noise. Stated at length because reporting 1.5× here would have been wrong by a third,
and nothing in the single-run table said so.

Two things fall out:

- **`save=False` buys nothing** (22.8 vs 22.9, inside the spread). `final_eval`'s
  second validation pass is not where the time goes, so the EMA-versus-raw-weights
  question it raises does not need answering. Dropped.
- **`plots` is the free lever.** It cannot change what is learned, and 1.19× is
  measured. `mosaic=0` is worth another 1.12× on top but changes the data path, so it
  is holdout-gated and stays a run-config key at its default.

---

## What to actually run

Ordered by expected value per GPU-hour on this machine, all of it gated behind the
phase-3 seed spread — nothing below is claimable until a difference bigger than
±0.016 mAP50 is what "better" means.

| # | Change | Where | Cost | Why |
|---|---|---|---|---|
| 0 ✅ | **`plots` on the final round only** | `server_app.py` + `client_app.py` ⚠ | done | **1.19×**, measured, and it cannot change what is learned. The pictures already only survived from the last round |
| 1 | `fraction_evaluate` as a run-config key, default < 1.0 | `server_app.py` ⚠ | one key | 13.8 % of wall clock, on a metric that is not the headline |
| 2 | `optimizer` set explicitly | `client_app.py` ⚠ | one key | until this lands, no LR experiment is running the LR it names |
| 3 | Server-driven `lr0` + `warmup_epochs`, broadcast per round | `server_app.py` + `client_app.py` ⚠ | small | safe to share one `FitIns`: the schedule is global, unlike the B9 `batch_id`. Attacks the 0.066 mAP50 round-1 loss |
| 4 | `mosaic` and `erasing` as run-config keys | `client_app.py` ⚠ | small | 30 % of the dataloader, and the dataloader is the bottleneck. Changes the data path, so holdout-gated |
| 5 | **FedBN** — BN tensors stay local | `get_set_model.py` + strategy ⚠ | medium | the only technique here aimed at *feature-shift* non-IID, which is the axis this fleet is partitioned on |
| 6 | True per-step FedProx | `DetectionTrainer` subclass ⚠ | ~15 lines | today's is a post-hoc weight-space pull, which is honest but is not FedProx |
| 7 | Personalised heads | strategy ⚠ | medium | stacks with 5; both are "share the backbone" |

### The one that is not on the list

`num_examples` weights FedAvg by **images**, not objects. `car` is 55.4 % of objects
and object density varies with condition — a night-city shard and an open-highway shard
of equal image count do not carry equal supervision. Whether the weight should be
objects instead of images is a real question for federated *detection* that does not
arise in the classification papers this design inherits from. It is not obviously
"yes": images is what the client can count cheaply and honestly, objects would let one
crowded vehicle dominate. Flagged here rather than filed as a change, because the
experiment that settles it is a phase-3-shaped one.

### Per-step FedProx, since the mechanism note is easy to get wrong

Re-confirmed against 8.4.115: `"optimizer_step"` **is** a key in
`default_callbacks`, and `BaseTrainer.optimizer_step` **never calls
`run_callbacks` for it** — the method body is `scaler.unscale_` → `clip_grad_norm_` →
`scaler.step` → `scaler.update` → `zero_grad` → `ema.update`, with no callback hook
anywhere. Registering that callback is a silent no-op that looks like it works.

`optimizer_step` is an ordinary method and `YOLO.train()` accepts `trainer=`, so the
proximal term goes in a subclass that adds `μ · (w − w_global)` to `p.grad` before
calling `super().optimizer_step()`, with the global weights captured at round start.

### And a clamp worth restating

`_get_warmup_iterations` is `min(warmup_epochs, max(epochs - 1, 0))`. At
`local_epochs = 1` there is **no warmup at all**; at 4 there really are three warmup
epochs of four. Any statement about warmup has to name the `local_epochs` it holds at.
