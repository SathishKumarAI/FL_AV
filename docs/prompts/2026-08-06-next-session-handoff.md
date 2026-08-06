# Handoff — start here next session

Paste this as the opening message. It exists so the next session spends its tokens on
work rather than on rediscovering the state of the project.

---

## Where things are

Branch `feat/pipeline-observability`, 41 commits ahead of `main`, all pushed. Off
`fix/gpu-bringup` (PR #32), off `main`. Working tree clean except two pre-existing
file-mode changes (`Research_docs/...`, `my-project/scripts/gen_certs.sh`).

**The project has a result now.** Both sides of the comparison made exactly
201 600 image-visits — 6 vehicles × 1 400 images × 6 rounds × 4 local epochs against
8 400 pooled images × 24 epochs — and parity is asserted in the artifact
(`pipeline/.state/baseline-8400img-24ep.json`: `"ratio": 1.0, "matched": true`).

| on 1 000 held-out images | federated | centralised | gap | retained |
|---|---|---|---|---|
| mAP50 | 0.4173 | **0.4936** | +0.0763 | **84.5 %** |
| mAP50-95 | 0.2313 | **0.2770** | +0.0457 | 83.5 % |

Federation costs ~15 % of the achievable accuracy at an identical budget, in exchange
for never pooling the data. The federated curve is monotonic over all six rounds
(0.3329 → 0.4173) and the aggregate checksum moved every round.

**Believe nothing smaller than ±0.016 mAP50.** An earlier ceiling with 1.667× the
data and compute scored *lower* (0.4771). That is the measured noise floor until seed
repeats exist.

## Read these first, in this order

1. `STATUS.md` — the result, the traps confirmed today, and the ranked next tasks
2. `CLAUDE.md` — hard rules, the change→file map, and the silent failures this project has shipped
3. `docs/RUNBOOK.md` — how to run and compare anything without an assistant
4. `docs/BACKLOG_100.md` — what to build, with 14 items marked done

## Run it

```powershell
.\scripts\run_pipeline.ps1                       # demo, ~10 min, proven end to end
.\scripts\run_pipeline.ps1 -Profile full -Vehicles 6 -PerVehicle 1400 -Rounds 6 -Epochs 4 -Baseline
python -m pipeline.server                        # six tabs on http://127.0.0.1:8800
```

Comparisons, one command each:

```bash
python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes
python -m pipeline.experiment --preset strategies --strategies fedavg,fedadam,fedavgm --yes
python -m pipeline.experiment --preset partitions --partitions condition,random,dirichlet --yes
python -m pipeline.experiment --preset alpha --alphas 0.05,0.5,100 --yes
python -m pipeline.ledger        # every run: approach, data, time, cost, result
python -m pipeline.compare --last 10 --md
```

## Do these next, in this order

### 1. ⚠ Absolute paths in my-project's loggers — small, unblocks trust in `verify`

`utils/logging_setup.py` configures `logs/server.log` **relative to the CWD, at import
time**. Importing `my_project.server_app` therefore writes an empty
`logs/server.<pid>.log` wherever the process happens to be standing — and
`pytest my-project/tests` does exactly that, at collection, from the repo root.

That empty file looked newer than the real federation's log and made `verify` report
`need >=2 rounds to tell, saw 0` minutes after a six-round run had succeeded. The
pipeline is now robust to it (`logparse.latest_run_log` only trusts a log that
aggregated a round), but the cause is still there. Own branch, own prompt.

### 2. ⚠ Backlog 30 — an LR schedule for short rounds — the highest-value ML change

The Metrics tab shows box, cls **and** dfl all *rising* across the four epochs of
every round: each client ends the round worse than the aggregate it started from.
Warmup is three epochs of a four-epoch round, so the schedule never leaves warmup.
This is the most likely reason 24 effective epochs stopped at 0.4173 while the
centralised model reached 0.4936 on the same budget.

Touches `client_app.py`. Own branch, own prompt. Verify with the per-epoch panel:
the losses should fall within a round afterwards.

### 3. Backlog 42 — seed repeats — the prerequisite for every other comparison

`python -m pipeline.experiment --preset seeds --seeds 0,1,2 --yes`, full profile.
Until the spread across repeats is known, no difference between two approaches means
anything. The Metrics tab already groups repeats and prints the spread.

### 4. Backlog 31 — rounds × epochs at constant product

12×2, 6×4, 3×8 at the same image-visits. Measures client drift directly, and item 2
predicts fewer local epochs will win.

### 5–7

Backlog 80 (MLflow — it refused this run: the file store is in maintenance mode and
needs a SQLite backend), backlog 36 (`car` is 55.4 % of all objects, `train` has 29
instances fleet-wide), and superseding the reports written before today's provenance
fixes — they mix runs in their stored `learning` block, are flagged in the Metrics
tab, and cannot be repaired retroactively.

## What exists now that did not this morning

| | Command |
|---|---|
| Shared holdout, carved before the fleet | `python -m pipeline.holdout --build` |
| Global model scored on it, per round | `python -m pipeline.holdout --evaluate` |
| Centralised ceiling at a matched budget | `python -m pipeline.baseline --rounds 6 --local-epochs 4` |
| Shard validation, six checks, read-only | `python -m pipeline.validate` |
| Dataset composition, classes and conditions | `python -m pipeline.dataset_stats` |
| The budget arithmetic before you spend it | `python -m pipeline.plan` |
| Every run as one comparable record | `python -m pipeline.ledger` |
| Dirichlet partitioning, α as the knob | `--partition dirichlet --alpha 0.3` |
| Any of 12 Flower strategies | `--strategy fedadam` |

Stage chain: env → dataset → populate → **holdout** → fleet → **validate** → sanity →
federate → **evaluate** → verify → **baseline**.

Dashboard tabs: **Control · Live · Data · Metrics · Plan · Docs**. The Docs tab is
generated from the modules' own docstrings, so it cannot describe code that has since
changed; a test asserts its tab list matches the markup.

## The rules that have not changed

1. `pipeline/` never modifies `my-project/` — enforced by a test.
2. No data, checkpoints, reports or credentials committed — enforced by a test.
3. Assemble before building: MLflow owns metrics, Ray owns actor internals.
4. Fail loudly. A failed stage halts the chain.
5. Verify the effect, not the absence of errors. Put the number in the commit message.

## The lesson this session actually taught

Twelve defects were found, and **seven of them only appear when the pipeline is driven
by a script rather than typed into a shell** — the way anyone reproducing the project
would run it. Four more were one root cause in different clothes: checkpoints,
checksums, `metrics.csv` and client logs all mixing runs, because the directory is
never cleared between them.

The most expensive one was a budget guard that computed its own reference from the
same wrong shard list it was checking, and printed `ratio 1.0` for a 1.5× advantage.
A check that derives its baseline from the thing it is checking is not a check.

So: run it end to end from a script before believing it works, and make every guard
take its reference from somewhere the thing under test cannot reach.

## Verification

```bash
python -m pytest pipeline/tests -q        # 118
python -m pytest my-project/tests -q      # 31
python -m pipeline.verify                 # the four pass criteria
python -m pipeline.validate               # the six shard checks
python -m pipeline.holdout --evaluate     # the honest global metric
```
