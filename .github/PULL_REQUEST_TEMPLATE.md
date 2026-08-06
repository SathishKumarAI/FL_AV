## What was wrong or missing

<!-- The problem, and how you know it is real. Numbers or log lines beat adjectives. -->

## What changed

<!-- And what you deliberately did NOT change, if a reviewer might expect it. -->

## How it was verified

<!-- The command and its ACTUAL output. Not "tests pass" -- the count.
     Not "it works" -- the metric that moved.

     This project has shipped changes that looked successful and did nothing:
     clients returning the weights they were sent, every client training the same
     shard, a round with no optimizer step. Verify the effect, not the absence of
     errors. -->

```
$ python -m pytest my-project/tests pipeline/tests -q

```

## Checklist

- [ ] A test fails without this change
- [ ] Full suite green
- [ ] CI green, including the federation smoke
- [ ] Docs updated in this PR (`README` / `docs/` / `CLAUDE.md` / `STATUS.md`)
- [ ] No data, checkpoints, reports or credentials staged
- [ ] `my-project/pyproject.toml` not committed in its flwr-rewritten form
- [ ] `STATUS.md` says where this leaves the project

## Still open

<!-- Known gaps, follow-ups, anything you chose to defer and why. -->
