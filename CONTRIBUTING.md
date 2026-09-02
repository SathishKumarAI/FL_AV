# Contributing — branches, commits, merges

The rules exist to keep `main` releasable and to make history worth reading a year from
now. They are short because rules nobody remembers are not rules.

## Branches

`main` is always green: tests pass, CI passes, and the federation runs. Nothing is
committed to it directly.

| Prefix | For | Example |
|---|---|---|
| `feat/` | new capability | `feat/pipeline-observability` |
| `fix/` | a defect with a reproduction | `fix/gpu-bringup` |
| `docs/` | documentation only | `docs/gpu-test-plan` |
| `exp/` | an experiment that may be thrown away | `exp/fedadam-sweep` |
| `chore/` | tooling, deps, CI | `chore/bump-flwr` |

One branch, one concern. A branch that fixes a bug *and* adds a feature is two branches
that have not been separated yet — and it makes the eventual revert impossible to scope.

`exp/` branches are allowed to be messy and are **not** required to be merged. Delete
them once their result is written down somewhere permanent.

## Commits

**A commit message explains the change; the diff already shows it.** Say what was wrong
and why this is the fix. If the reason lives only in your head, the commit is
incomplete.

```
<type>: <what changed, imperative, lower case>

Why it was wrong, and what evidence says the fix works. Include the numbers:
"eval mAP50 0.275 -> 0.320", "24 tests pass", "checksums differ between rounds".
Name the trap if there was one -- the next person will hit it too.
```

Types: `feat` `fix` `docs` `test` `chore` `perf` `refactor`.

Rules that have earned their place here:

- **Never commit generated data.** Datasets, checkpoints, MLflow stores, reports, run
  artifacts. If a change can produce a new kind of artifact, its ignore rule goes in the
  **same commit**. A test asserts the rules still match real paths.
- **Never commit credentials.** Nothing here needs any.
- **Never commit `my-project/pyproject.toml` after a `flwr run`.** flwr rewrites it in
  place, commenting out `[tool.flwr.federations]`. Committing that leaves a fresh clone
  with no federation to run — which has already happened once. The pipeline restores it
  automatically; check `git diff` before staging.
- **Stage explicitly.** `git add <paths>`, not `git add -A`, in a repo whose working
  tree routinely contains gigabytes of run output.
- **Green before commit.** `python -m pytest my-project/tests pipeline/tests -q`.

## Pull requests

Every change reaches `main` through a PR, including your own. The PR body is where the
*reasoning* goes; use the template.

A PR should answer, without the reviewer opening the diff:

1. What was broken or missing, and how you know.
2. What you changed, and what you deliberately did **not**.
3. How it was verified — the command and its actual output.
4. What is still open.

**Squash-merge** feature branches: one commit on `main` per landed concern, so `main`'s
history is a list of decisions rather than a list of keystrokes. Keep the merge title as
the conventional-commit line.

Delete the branch after merge. Stale branches are how people end up reviewing code that
was superseded three days ago.

## Reviews

- Reviewing your own PR is allowed here (small team) but read it as a stranger would.
- A review that only says "looks good" costs more than it saves. Say what you checked.
- If CI is red, the PR is not ready — do not merge and "fix forward".

## Definition of done

A change is done when all of these are true. Not most.

- [ ] Tests cover the failure mode that motivated it — the test fails without the fix
- [ ] Full suite green: `pytest my-project/tests pipeline/tests -q`
- [ ] CI green, including the end-to-end federation smoke
- [ ] Docs updated in the same PR — `README`, `docs/`, `CLAUDE.md`, `STATUS.md`
- [ ] No data, artifacts, or credentials staged
- [ ] `STATUS.md` says where the work stopped and what is next

## The rule behind the rules

This project has shipped several changes that **looked** successful and did nothing:
clients returning the weights they were sent, every client training the same shard, a
round completing without an optimizer step, a report that failed silently. All of them
passed a casual "it ran without errors" check.

So: **verify the effect, not the absence of errors.** Paste the number into the commit.
