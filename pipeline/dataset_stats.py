"""What the fleet is actually training on, as numbers rather than a claim.

The dashboard could show one vehicle's mixture on demand; nothing showed the fleet as
a dataset. This reads the shards and the label files and answers: how many images
exist, how the 13 classes are distributed, which conditions are thin, and whether the
holdout looks like the data it is meant to measure.

Reading 14 000 label files takes seconds, which is far too slow for a 2-second poll,
so the result is cached against the fleet fingerprint: the same fleet is computed
once, and a rebuilt fleet invalidates itself.

    python -m pipeline.dataset_stats            # the same numbers, on the console
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import holdout, paths, vehicles

CACHE = paths.STATE / "dataset_stats.json"


def class_names() -> list[str]:
    """From a real data.yaml, so the labels and the names cannot drift apart."""
    for candidate in (paths.VEHICLE_BATCHES / "batch_1" / "data.yaml",
                      paths.PROJECT / "batch" / "batch_1" / "data.yaml"):
        if candidate.exists():
            names = []
            for line in candidate.read_text().splitlines():
                if line.startswith("- "):
                    names.append(line[2:].strip())
            if names:
                return names
    return [str(i) for i in range(13)]


def class_histogram(label_dir: Path) -> dict[str, int]:
    """class id -> instance count, straight from the label files.

    Only the first field of each row is read: this runs over tens of thousands of
    files and the box coordinates are not the question being asked.
    """
    counts: dict[str, int] = {}
    if not label_dir.is_dir():
        return counts
    with os.scandir(label_dir) as it:
        for entry in it:
            if not entry.name.endswith(".txt"):
                continue
            try:
                with open(entry.path, "r") as fh:
                    for row in fh:
                        head = row.split(" ", 1)[0].strip()
                        if head:
                            counts[head] = counts.get(head, 0) + 1
            except OSError:
                continue
    return counts


def _mix(names: list[str], index: dict) -> dict[str, dict[str, int]]:
    out = {"weather": {}, "scene": {}, "timeofday": {}}
    for name in names:
        attrs = index.get(name) or {}
        for key, bucket in out.items():
            value = attrs.get(key) or "unknown"
            bucket[value] = bucket.get(value, 0) + 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in out.items()}


def _listing(path: Path) -> list[str]:
    return [n.strip() for n in path.read_text().splitlines() if n.strip()] if path.exists() else []


def compute() -> dict:
    """Everything the Data tab shows. Seconds, not milliseconds -- hence the cache."""
    index = vehicles.cached_attributes()
    names = class_names()
    fleet = vehicles.load_fleet()
    meta = vehicles.load_fleet_meta()
    held = holdout.names()

    shards = []
    fleet_classes: dict[str, int] = {}
    for entry in fleet:
        vid = entry.get("vid")
        root = paths.VEHICLE_BATCHES / f"batch_{vid}"
        train = _listing(root / "train.txt")
        classes = class_histogram(root / "labels" / "train")
        for k, v in classes.items():
            fleet_classes[k] = fleet_classes.get(k, 0) + v
        shards.append({
            "vid": vid,
            "condition": entry.get("condition"),
            "n_train": entry.get("n_train"),
            "n_val": entry.get("n_val"),
            "fingerprint": entry.get("fingerprint"),
            "labels": sum(classes.values()),
            "classes": classes,
            "mix": _mix(train, index),
            "held_out_inside": sum(1 for n in train if n in held),
        })

    pool = paths.find_pool()
    holdout_train = sorted(held)
    holdout_classes = class_histogram(holdout.HOLDOUT_ROOT / "labels" / "val")

    return {
        "class_names": names,
        "fleet": shards,
        "fleet_classes": fleet_classes,
        "fleet_meta": meta,
        "holdout": {
            "size": len(held),
            "meta": holdout.meta(),
            "classes": holdout_classes,
            "labels": sum(holdout_classes.values()),
            "mix": _mix(holdout_train, index),
        },
        "pool": {
            "val_images": paths.count_files(pool / "val", {".jpg"}) if pool else 0,
            "train_images": paths.count_files(pool / "train", {".jpg"}) if pool else 0,
            "indexed": len(index),
            "path": str(pool) if pool else None,
        },
        "pooled_baseline": paths.count_files(
            paths.VEHICLE_ROOT / "pooled" / "images" / "train", {".jpg"}),
    }


def cached(force: bool = False) -> dict:
    """`compute()`, memoised against the fleet fingerprint.

    A rebuilt fleet has a different fingerprint and invalidates this by itself, which
    is the property a mtime or a TTL would not give.
    """
    fingerprint = (vehicles.load_fleet_meta() or {}).get("fingerprint")
    if not force and CACHE.exists():
        try:
            data = json.loads(CACHE.read_text())
            if data.get("fingerprint") == fingerprint:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    data = {"fingerprint": fingerprint, **compute()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)

    d = cached(force=args.refresh)
    names = d["class_names"]
    total = sum(d["fleet_classes"].values()) or 1

    print(f"pool: {d['pool']['val_images']} val + {d['pool']['train_images']} train images, "
          f"{d['pool']['indexed']} indexed with attributes")
    print(f"fleet: {len(d['fleet'])} shards, fingerprint {d.get('fingerprint')}")
    print(f"holdout: {d['holdout']['size']} images, {d['holdout']['labels']} labelled objects\n")

    print("class distribution across the fleet:")
    for cid, count in sorted(d["fleet_classes"].items(), key=lambda kv: -kv[1]):
        label = names[int(cid)] if cid.isdigit() and int(cid) < len(names) else cid
        print(f"  {label:<16} {count:>8}  {100 * count / total:5.1f}%")

    print("\nper shard:")
    for s in d["fleet"]:
        leak = f"  LEAK {s['held_out_inside']}" if s["held_out_inside"] else ""
        print(f"  v{s['vid']:<3} {str(s['condition'])[:26]:<28} {s['n_train']:>6} img "
              f"{s['labels']:>7} objs  {s['fingerprint']}{leak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
