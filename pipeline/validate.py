"""Check the shards before believing anything measured on them.

Every number this project produces is downstream of the fleet's shards, and each way
they can be wrong is silent: an image with no label trains as background, an image in
two vehicles gets double the vote in FedAvg's weighted average, an image in both train
and val turns evaluation into a memory test, and a held-out image inside a shard makes
the one honest metric self-referential.

All of those produce a run that completes and reports plausible numbers.

Read-only by design. It names what is wrong and refuses to repair it: a validator that
quietly fixes data hides the bug that produced the data.

    python -m pipeline.validate
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import holdout, paths

SAMPLE = 5          # offenders to name per problem; the count carries the rest


@dataclass
class Problem:
    check: str
    count: int
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        shown = ", ".join(self.examples[:SAMPLE])
        more = f" (+{self.count - len(self.examples[:SAMPLE])} more)" if self.count > SAMPLE else ""
        return f"{self.check}: {self.count} — {shown}{more}"


def _listing(shard: Path, split: str) -> list[str]:
    f = shard / f"{split}.txt"
    if not f.exists():
        return []
    return [n.strip() for n in f.read_text().splitlines() if n.strip()]


def _present(shard: Path, split: str) -> set[str]:
    d = shard / "images" / split
    if not d.is_dir():
        return set()
    with os.scandir(d) as it:
        return {e.name for e in it if e.is_file()}


def _label_rows_look_wrong(path: Path, nc: int) -> str | None:
    """Return why a label file is unusable, or None. Cheap: no image decoding."""
    try:
        text = path.read_text()
    except OSError as e:
        return f"unreadable ({e.__class__.__name__})"
    rows = [r for r in text.splitlines() if r.strip()]
    if not rows:
        return "empty"
    for row in rows:
        fields = row.split()
        if len(fields) < 5:
            return f"row has {len(fields)} fields, expected 5+"
        try:
            cls = int(float(fields[0]))
            coords = [float(v) for v in fields[1:5]]
        except ValueError:
            return "non-numeric field"
        if not 0 <= cls < nc:
            return f"class id {cls} outside 0..{nc - 1}"
        if any(not 0.0 <= c <= 1.0 for c in coords):
            return "coordinates outside 0..1 (not normalised)"
    return None


def check_fleet(root: Path | None = None, nc: int = 13, held: set[str] | None = None,
                deep_labels: bool = True) -> list[Problem]:
    """Every way a fleet can be quietly wrong. Empty list means it is sound."""
    root = root or paths.VEHICLE_BATCHES
    held = holdout.names() if held is None else held
    problems: list[Problem] = []

    shards = sorted(root.glob("batch_*")) if root.is_dir() else []
    if not shards:
        return [Problem("no fleet on disk", 1, [str(root)])]

    missing_label, missing_image, bad_label = [], [], []
    train_owner: dict[str, str] = {}
    duplicate_train, split_overlap, in_holdout = [], [], []

    for shard in shards:
        train, val = _listing(shard, "train"), _listing(shard, "val")
        both = set(train) & set(val)
        split_overlap += [f"{shard.name}/{n}" for n in sorted(both)]

        for split, names in (("train", train), ("val", val)):
            present = _present(shard, split)
            label_dir = shard / "labels" / split
            for name in names:
                if name not in present:
                    missing_image.append(f"{shard.name}/{split}/{name}")
                    continue
                label = label_dir / f"{name.rsplit('.', 1)[0]}.txt"
                if not label.exists():
                    missing_label.append(f"{shard.name}/{split}/{name}")
                elif deep_labels:
                    why = _label_rows_look_wrong(label, nc)
                    if why:
                        bad_label.append(f"{shard.name}/{split}/{label.name} ({why})")
                if name in held:
                    in_holdout.append(f"{shard.name}/{split}/{name}")

        for name in train:
            owner = train_owner.get(name)
            if owner and owner != shard.name:
                duplicate_train.append(f"{name} in {owner} and {shard.name}")
            else:
                train_owner[name] = shard.name

    for check, offenders in (
        ("images listed but not materialised", missing_image),
        ("images with no label file", missing_label),
        ("unusable label files", bad_label),
        ("images shared between two vehicles' train sets", duplicate_train),
        ("images in both train and val of one shard", split_overlap),
        ("held-out images found inside a shard", in_holdout),
    ):
        if offenders:
            problems.append(Problem(check, len(offenders), offenders[:SAMPLE]))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", type=int, default=13)
    ap.add_argument("--fast", action="store_true",
                    help="skip reading label contents; check only that they exist")
    args = ap.parse_args(argv)

    problems = check_fleet(nc=args.nc, deep_labels=not args.fast)
    shards = sorted(paths.VEHICLE_BATCHES.glob("batch_*")) if paths.VEHICLE_BATCHES.is_dir() else []
    print(f"validating {len(shards)} shards under {paths.VEHICLE_BATCHES}")
    held = holdout.names()
    print(f"holdout: {len(held)} images that must not appear in any of them")

    if not problems:
        print("\nOK — labels present and usable, slices disjoint, splits separate, "
              "holdout intact.")
        return 0

    print("\nPROBLEMS — every metric measured on these shards is suspect:\n")
    for p in problems:
        print(f"  {p}")
    print("\nNothing was repaired. Rebuild the fleet (`python -m pipeline.build_fleet`) "
          "or fix the source data; a validator that edits your data hides the bug that "
          "produced it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
