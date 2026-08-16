# Prompt — make the repo survive contact with a machine that is not this one

**Branches:** `chore/oss-polish-and-nightly` (#35), `fix/logger-paths-anchored-to-the-project`
(#36), `fix/tests-that-only-passed-in-a-used-checkout` (#37, merged),
`build/cpu-reproduction-container-v2` (#40) · **Written:** 2026-08-16

## The brief

Four separate asks, one theme: everything here has only ever been true on one laptop.

- **Open-source polish** — the repo declares a licence it does not ship.
- **CI / phase G** — `main` is green at merge time and never checked again.
- **Packaging** — a fresh clone costs an hour of environment traps before a single
  test runs.
- **Ops hardening** — logs land wherever the process happened to start.

One increment per branch, each independently mergeable, each with a runnable check.

## What was actually found

The brief was written from a list of gaps. Three of the four turned out to be worse
than described, and the fourth turned out to be the wrong diagnosis entirely.

### `pyproject.toml` has always declared a licence that does not exist

`license = "Apache-2.0"` since the first commit; no `LICENSE` file, ever. A declaration
in a build manifest is not a grant, so the repo was legally all-rights-reserved while
advertising otherwise. Fixed by shipping the verbatim text, not by editing the claim.

### `pytest pipeline/tests` was red on every clean clone

Green here, red anywhere else — which is what CI is. Three tests borrowed state that
only exists on a machine that has already used the project:

| Test | Borrowed |
|---|---|
| `test_the_sanity_stage_does_not_shell_out_to_a_moving_target` | a built fleet on disk |
| `test_the_report_leads_with_the_metric_no_client_could_flatter` | a completed run's `metrics.csv` |
| `test_generated_paths_are_all_gitignored` | a `.git` **and** a `git` binary |

None of the assertions were weakened; each test was given the state it had been
silently taking from a developer's disk. Found by running the suite in a fresh
worktree, then in a container — not by reading it.

### `subprocess_env` did not do what its own comment says

```python
scripts = str(Path(sys.executable).parent)
if scripts not in env.get("PATH", "").split(os.pathsep):
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
```

The comment promises the interpreter's Scripts directory "goes first on PATH". The code
skips the prepend whenever it is already *anywhere* on PATH. Present is not first, and
only first is the guarantee that matters: flwr resolves `flower-superlink` from PATH
rather than from its own location, and our children spawn children. On a GitHub Windows
runner `setup-python` puts the directory behind PowerShell's, so **every CI run had been
executing with the ordering silently absent**. Caught by a Windows-only test failure.

### The container is a diagnostic tool, not a deployment

`docker run --rm federated-yolov8:cpu` → 149 passed, 1 skipped. It cannot train (CPU
torch, no GPU passthrough), has no data (BDD100K is 7.6 GB in a kagglehub cache and the
first hard rule here is that data is never committed), and is not a deployment. Its
value is that it runs the suite somewhere that has never been used — which is what found
the three tests above.

`.dockerignore` is written as a **safety rule rather than an optimisation**: the build
context is the one place the never-commit-data rule can be bypassed without touching
git. Verified from inside the image: 0 `.jpg`, 0 `.pt`.

## The one that was diagnosed wrong, twice, and why it matters

`configure_logging("server", "logs/server.log")` runs at import time in five modules,
so the log's location was decided by whichever directory the process started in. Merely
importing `my_project.server_app` — which `pytest` does at collection — dropped an empty
`logs/server.<pid>.log` wherever you were standing, and that file once looked newer than
a real federation's log and made `verify` report "need >=2 rounds to tell, saw 0" right
after a six-round run had succeeded.

**Attempt 1** anchored on `Path(__file__).resolve().parents[1]`. The CI smoke then
failed. The cause looked obvious, because one line in the job output says:

```
Successfully installed my-project to /home/runner/.flwr/apps/flower.my-project.1.0.0.480fd449
```

`flwr run` does not execute the checkout — it copies the app and runs the copy. So
`__file__` pointed into a cache and the logs went where nobody would look.

**Attempt 2** made `project_root()` prefer `FL_AV_DATA_ROOT` (which the pipeline and CI
both set), fall back to `__file__`, and refuse a path under `.flwr`. The smoke failed
again, identically.

**Attempt 3 was not needed, because the premise was wrong.** After the branch was
rebased, the same code passed the smoke. The shards drawn tell the story:

| Run | Shards | Result |
|---|---|---|
| 31963218802 | batch3, batch10 | fail |
| 31964432771 | batch3, batch8 | fail |
| 31966054382 | batch3, batch9 | **pass** |

Same branch, same code, different shards, different answer — and `batch3` is in all
three, so it is not the shard itself. **The smoke is non-deterministic** because
`server_app.py` assigns shards with an unseeded `random.choice` over a hardcoded range
of 1–10 (issue #41). Two rounds of "fix" went into a change that was never the cause.

The `flwr`-runs-a-copy finding stands on its own and is worth keeping. The attribution
of the smoke failure to the logging change does not.

## What that cost, and the cheap thing that would have prevented it

Flower reports a client exception as a **count** — `aggregate_fit: received 0 results
and 2 failures` — and writes the traceback into the client's own log file, which the
smoke job had never printed. So the number saying the run was broken arrived with the
artifact saying why stripped out.

Four six-minute CI rounds went into inferring a cause from a failure count and a
wall-clock duration. The fix is one `if: failure()` step that dumps the client logs, and
it has to sit **after** the assert, because `failure()` only reports on steps that have
already run.

See [`CI_TRAPS.md`](../CI_TRAPS.md) for that and the three other ways CI misled this
session.

## Verification

| | |
|---|---|
| `pytest my-project/tests pipeline/tests -q` | 154 passed |
| `pytest pipeline/tests -q` in a fresh worktree, no fleet, no run history | 118 passed |
| CI-equivalent: `python:3.12-slim`, pytest + pyyaml only, no flwr, no git | 118 passed, 1 skipped |
| `docker run --rm federated-yolov8:cpu` | 149 passed, 1 skipped |
| Mutation — old `if scripts not in PATH` logic restored | new ordering test **fails** |
| Mutation — `from ultralytics import YOLO` → `ultralytics.cfg` | sanity test **fails** |
| Mutation — `project_root()` reduced to `__file__` only | both new logging tests **fail** |
| All four YAML files parse; merged `ci.yml` keeps all three jobs | ✓ |

## Deliberately not done

- **Branch protection** — PR required, CI required, squash-only, delete on merge. A
  repository setting rather than a file, and the owner's to make.
- **`CODE_OF_CONDUCT.md`** — a solo research repo with no contributors does not need one
  to be honest, and an unenforced one is worse than none.
- **Building the image in CI** — so it can rot. The obvious follow-up, left out to keep
  the container PR to one thing.
- **The federation smoke inside the container** — it needs the `laptop_copy` fixture
  fetched at build time, which puts a git dependency into an image that deliberately has
  no `.git`. That gap is exactly how the misdiagnosis above survived as long as it did.
