# Prompt — validate the shards before believing anything measured on them

Written before the code. Backlog 66 and 74.

## The problem

Every number this project produces is downstream of the shards, and nothing checks
them. The failure modes are all silent:

- **An image without a label** trains as a background image. Ultralytics accepts it,
  reports nothing, and the vehicle quietly learns that its condition contains no
  objects.
- **The same image in two vehicles** trains twice per round and is counted twice in
  `num_examples`, which is FedAvg's aggregation weight — so one image gets double the
  vote.
- **An image in both train and val** turns a vehicle's self-evaluation into a memory
  test, and its mAP into a number that means nothing.
- **A holdout image inside a shard** makes the one honest metric partly
  self-referential, which is the whole reason the holdout was carved.
- **An empty or truncated label file** reads as "no objects here" rather than as the
  corruption it is.

Each of these produces a run that completes, reports plausible metrics, and is wrong
— the exact pattern in this repo's silent-failure table.

## What to build

`pipeline/validate.py`, and a `validate` stage between `fleet` and `sanity`:

| Check | Fails when |
|---|---|
| label coverage | an image in a shard's split list has no label file |
| listing integrity | `train.txt`/`val.txt` name a file that is not materialised |
| cross-shard leakage | one image appears in two vehicles' train sets |
| split leakage | one image appears in both train and val, in any shard |
| holdout containment | a held-out image appears in any shard |
| label sanity | a label file is empty, or has a row that is not 5 numeric fields with a class id inside `nc` |

Report every failure with counts and the first few offenders, not just the first one:
finding out about the second problem after fixing the first is how an evening goes.

## Non-negotiable

- **Read-only.** It inspects shards; it never repairs them. A validator that silently
  fixes data hides the bug that produced it.
- Fails loudly and halts the chain, like any other stage.
- Fast enough to run every time: no image decoding, no label parsing beyond the field
  check, and a single directory scan per shard.

## Verification

- `python -m pytest pipeline/tests -q` — each failure mode has a test that builds a
  broken shard in `tmp_path` and asserts the specific complaint.
- `python -m pipeline.validate` against the real ten-shard fleet on disk prints a
  clean bill or names what is wrong.
