#!/usr/bin/env python3
"""Populate batch/batch_N/images/{train,val} from a BDD100K image pool.

The repo commits the YOLO *labels* and the split lists (train.txt/val.txt) but
not the JPEGs, so every shard is unrunnable until the images are linked in.

Hardlinks by default (near-zero disk, pool and repo must be on the same volume);
falls back to copy. Also drops the stale `labels/*.cache` files, which were built
on another machine and will make Ultralytics skip the scan and fail.

    python scripts/populate_images.py --pool D:/bdd100k/images/100k
    python scripts/populate_images.py --pool ... --batches 1,2 --limit 200  # smoke shard
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

SPLITS = ("train", "val")


def index_pool(pool: Path, split: str) -> dict:
    """Map basename -> path for every JPEG under <pool>/<split>/, recursively.

    Mirrors disagree on layout: the Kaggle copy of BDD100K nests its train split
    into `train/trainA/`, `train/trainB/`, ... while the official archive is flat.
    A flat-only lookup silently finds ~2% of the train set, so index by basename
    and let the directory shape be whatever it is.
    """
    base = pool / split
    if not base.is_dir():
        return {}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return {p.name: p for p in base.rglob("*") if p.suffix.lower() in exts}


def link_split(batch_dir: Path, pool: Path, split: str, limit: int, copy: bool,
               index: dict = None):
    """Link the images named in <batch>/<split>.txt into <batch>/images/<split>/."""
    listing = batch_dir / f"{split}.txt"
    if not listing.exists():
        return 0, 0, [f"missing {listing}"]

    names = [n.strip() for n in listing.read_text().splitlines() if n.strip()]
    if limit:
        names = names[:limit]

    if index is None:
        index = index_pool(pool, split)

    dest = batch_dir / "images" / split
    dest.mkdir(parents=True, exist_ok=True)

    linked = skipped = 0
    missing = []
    for name in names:
        target, src = dest / name, index.get(name)
        if target.exists():
            skipped += 1
            continue
        if src is None or not src.exists():
            missing.append(name)
            continue
        # ponytail: hardlink, copy fallback. Symlinks need admin/dev-mode on Windows.
        try:
            if copy:
                raise OSError("copy requested")
            os.link(src, target)
        except OSError:
            shutil.copy2(src, target)
        linked += 1

    # Stale scan caches point at the previous machine's paths -> delete.
    cache = batch_dir / "labels" / f"{split}.cache"
    if cache.exists():
        cache.unlink()

    return linked, skipped, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="dir holding train/ and val/ BDD100K JPEGs")
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1] / "batch",
                    help="batch root (default: my-project/batch)")
    ap.add_argument("--batches", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--limit", type=int, default=0, help="images per split, 0 = all")
    ap.add_argument("--copy", action="store_true", help="copy instead of hardlink")
    args = ap.parse_args(argv)

    pool, root = Path(args.pool), Path(args.root)
    if not pool.is_dir():
        sys.exit(f"pool not found: {pool}")

    # Index each split once; rglob over 70 k files per shard would be 10x the work.
    indexes = {}
    for split in SPLITS:
        indexes[split] = index_pool(pool, split)
        print(f"pool/{split}: {len(indexes[split])} images")

    total_missing = 0
    for bid in [b.strip() for b in args.batches.split(",") if b.strip()]:
        batch_dir = root / f"batch_{bid}"
        if not batch_dir.is_dir():
            print(f"batch_{bid}: SKIP (no {batch_dir})")
            continue
        for split in SPLITS:
            linked, skipped, missing = link_split(batch_dir, pool, split, args.limit,
                                                  args.copy, indexes[split])
            total_missing += len(missing)
            note = f" MISSING={len(missing)} (e.g. {missing[0]})" if missing else ""
            print(f"batch_{bid}/{split}: linked={linked} present={skipped}{note}")

    if total_missing:
        print(f"\n{total_missing} images not in pool -- wrong pool dir, or a partial download.")
    return 1 if total_missing else 0


def _self_check():
    """assert-based check of link_split against a temp fixture."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pool = tmp / "pool" / "train"
        pool.mkdir(parents=True)
        (pool / "a.jpg").write_bytes(b"x")
        # Nested exactly like the Kaggle mirror's train/trainA, train/trainB, ...
        (pool / "trainA").mkdir()
        (pool / "trainA" / "nested.jpg").write_bytes(b"y")
        batch = tmp / "batch_1"
        (batch / "labels").mkdir(parents=True)
        (batch / "train.txt").write_text("a.jpg\nnested.jpg\ngone.jpg\n")
        (batch / "labels" / "train.cache").write_bytes(b"stale")

        assert set(index_pool(tmp / "pool", "train")) == {"a.jpg", "nested.jpg"}

        linked, skipped, missing = link_split(batch, tmp / "pool", "train", 0, False)
        assert (linked, skipped, missing) == (2, 0, ["gone.jpg"]), (linked, skipped, missing)
        assert (batch / "images" / "train" / "a.jpg").exists()
        # A subdirectory in the pool must not become a subdirectory in the shard.
        assert (batch / "images" / "train" / "nested.jpg").exists(), "nested pool image not linked flat"
        assert not (batch / "labels" / "train.cache").exists(), "stale cache not removed"

        # rerun is idempotent
        linked, skipped, _ = link_split(batch, tmp / "pool", "train", 0, False)
        assert (linked, skipped) == (0, 2), (linked, skipped)
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        sys.exit(main())
