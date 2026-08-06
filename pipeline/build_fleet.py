"""Materialise the simulated fleet. Invoked as a stage; usable standalone.

    python -m pipeline.build_fleet --vehicles 6 --per-vehicle 300
"""
from __future__ import annotations

import argparse
import json

import yaml

from . import paths, vehicles


def class_names() -> tuple[list[str], int]:
    """Read the class list from a source shard so the fleet cannot drift from it."""
    src = paths.PROJECT / "batch" / "batch_1" / "data.yaml"
    data = yaml.safe_load(src.read_text())
    return list(data.get("names", [])), int(data.get("nc", 13))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vehicles", type=int, default=6)
    ap.add_argument("--per-vehicle", type=int, default=300)
    ap.add_argument("--val-per-vehicle", type=int, default=0, help="0 = per-vehicle/5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    names, nc = class_names()
    index = vehicles.build_attribute_index()
    print(f"attribute index: {len(index)} images")

    # Build a shard for EVERY id in my-project's DEFAULT_BATCH_ID_RANGE, not just
    # --vehicles of them. The server picks ids at random from that whole range and
    # cannot be told otherwise without editing server_app.py, so a 6-vehicle run was
    # handed batch_id=7 and died on a directory that did not exist. --vehicles now
    # controls how many clients run; the addressable fleet is always the full range.
    n_shards = max(args.vehicles, len(list(paths.BATCH_IDS)))
    fleet = vehicles.assign(n_shards, args.per_vehicle,
                            val_per_vehicle=args.val_per_vehicle, seed=args.seed, index=index)
    root = vehicles.materialise(fleet, names, nc)

    for v in fleet:
        used = " (in this run)" if v.vid <= args.vehicles else ""
        print(f"  vehicle {v.vid}: {v.condition:<22} train={v.n_train:<6} val={len(v.val)}{used}")
    print(f"fleet root: {root}  (point FL_AV_DATA_ROOT here)")
    print(json.dumps([v.to_summary() for v in fleet]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
