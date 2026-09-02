# CI traps: four ways the pipeline lied about itself

Written 2026-08-16, after a session in which **more time went into misreading CI than
into writing the code it was checking**. `CLAUDE.md` already lists the silent failures
this project has shipped in its *training* stack. This is the same list for its
*continuous integration*, because the failure mode is identical: something reports a
result that is not the result.

Each entry is: what it looked like, what it was, and the one command that would have
answered it.

---

## 1. A conflicting PR silently runs no CI at all

**Looked like:** GitHub Actions was broken. A push landed, the Labeler ran, and the CI
workflow simply did not appear — no run, no failure, no "invalid workflow file"
annotation. Closing and reopening the PR did not help. Twice.

**Was:** the PR's base branch had moved underneath it, and the branch now conflicted on
`STATUS.md`. GitHub builds a **merge ref** for `pull_request` events; when the merge
conflicts, that ref cannot exist, so every `pull_request` workflow is skipped **without
a diagnostic**. The Labeler kept running only because it is `pull_request_target`, which
uses the base ref and needs no merge.

**One command:**

```bash
gh pr view <n> --json mergeable,mergeStateStatus
# mergeable=CONFLICTING  state=DIRTY   <- this, not infrastructure
```

**Rule.** When a run does not appear, check mergeability *before* forming any theory
about the runner, the quota, or the workflow file. A stacked PR whose base has just
merged is the common case, and it goes quiet rather than red.

**Two wrong theories were published before that command was run** — exhausted Actions
minutes, then a scheduling delay. Neither was ever checked.

---

## 2. Flower reports a client crash as a number, and files the reason elsewhere

**Looked like:** the federation ran and learned nothing.

```
aggregate_fit: received 0 results and 2 failures
Run finished 2 round(s) in 14.26s          # a healthy run takes 5m47s
AssertionError: expected >=2 aggregate checksums, got []
```

**Was:** each client raised, Flower caught it, counted it, and wrote the traceback into
that client's own log file — which the smoke job had never printed. The one number
saying the run was broken arrived with the one artifact saying why removed.

**Cost:** four six-minute CI rounds spent inferring a cause from a failure count and a
wall-clock duration, and a fix shipped twice for a bug that was never there (see trap 3).

**Fix, now in `ci.yml`:** an `if: failure()` step that dumps `logs/client.*.log` and
`logs/server.*.log`. It must sit **after** the assert step — `failure()` reports on steps
that have already run, so placed earlier it can only ever fire for a crash in `flwr run`
itself, never for the assertion it exists to explain.

---

## 3. The federated smoke is non-deterministic, so a failure proves nothing

**Looked like:** a change broke every client's `fit()`. The evidence was clean: the base
passed the smoke, base + change failed it, twice, identically.

**Was:** coincidence. `server_app.py` assigns shards with

```python
batch_id = random.choice(sorted(available_ids))   # unseeded, range hardcoded (1, 10)
```

and `random` is seeded nowhere in `my_project/`. The same branch, unchanged, then passed:

| Run | Shards drawn | Result |
|---|---|---|
| 31963218802 | batch3, batch10 | fail |
| 31964432771 | batch3, batch8 | fail |
| 31966054382 | batch3, batch9 | **pass** |

`batch3` is in all three, so the shard is not the cause either — but the outcome moves
with the draw, and that is enough to know the job cannot attribute blame.

**Rule.** Before concluding "my change broke CI", establish that the job is
deterministic. A flaky guard does not just miss bugs; it **manufactures** them, and a
confident wrong diagnosis costs more than no diagnosis.

Tracked as issue #41. Until it is seeded, a red smoke means *look*, never *revert*.

---

## 4. Reasoning about CI's environment is not measuring it

Three things were true of CI and not of the laptop, and each was discovered only by
being wrong first:

| | |
|---|---|
| The pipeline job installs **pytest and pyyaml only** | two tests needed `flwr` merely to assert about arguments, and had never run in CI |
| `python:3.12-slim` ships **without git** | a `.git`-existence guard was not enough; the binary has to exist too |
| `flwr run` executes a **copy** under `~/.flwr/apps/`, not the checkout | anything anchored on `__file__` resolves inside a cache |

The last one is worth restating on its own, because it is not a CI quirk — it is true of
every `flwr run` anywhere:

```
Successfully installed my-project to /home/runner/.flwr/apps/flower.my-project.1.0.0.480fd449
```

**Rule.** `docker run --rm -v "$PWD":/repo -w /repo python:3.12-slim bash -c "pip install
-q pytest pyyaml && pytest pipeline/tests -q"` reproduces the pipeline job in seconds.
Use it before pushing, not after CI disagrees.

---

## The shape all four share

Every one reported *something*, and the something was not the thing. A skipped workflow
reads as broken infrastructure; a failure count reads as a broken change; a flaky job
reads as a regression; a green laptop reads as a green repo.

The project's own rule already covers it — **measure, do not infer** — and this session
is the evidence that it applies to the tooling and not only to the model.
