"""The shared holdout: one val set no vehicle trains or self-evaluates on.

Every number this project reported until now came from a client scoring itself on
its own val split. Vehicle 3 trains on rain and is measured on rain; vehicle 1
trains on daytime city and is measured on daytime city. Those numbers cannot be
compared with each other, and averaging them is not a global metric -- it is a mean
over different distributions.

This module carves a fixed slice out of the val pool **before** the fleet is
assigned, materialises it once, and scores the global checkpoints on it out of
band. Out of band matters: server-side evaluation lives in ``my-project`` and this
component is not allowed to modify it, but the checkpoints are already on disk and
Ultralytics can read them without the federation knowing.

    python -m pipeline.holdout --build --size 1000
    python -m pipeline.holdout --evaluate
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import paths, vehicles

HOLDOUT_ROOT = paths.VEHICLE_ROOT / "holdout"          # gitignored with the rest
NAMES_FILE = paths.STATE / "holdout.json"
METRICS_FILE = paths.STATE / "holdout_metrics.json"


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select(size: int = 1000, seed: int = 0, val_pool: set[str] | None = None) -> list[str]:
    """Pick the holdout deterministically from the val pool.

    Sorted before shuffling on purpose: ``set`` iteration order varies between
    processes, so shuffling the raw set would give a different holdout on every
    machine and make two runs incomparable without either of them looking wrong.
    """
    pool = sorted(vehicles._available(Path("val")) if val_pool is None else val_pool)
    if len(pool) < size:
        raise SystemExit(f"holdout needs {size} val images, the pool has {len(pool)}. "
                         f"Run the populate stage, or ask for fewer.")
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:size])


def names() -> set[str]:
    """The current holdout, or an empty set if none has been carved yet."""
    if not NAMES_FILE.exists():
        return set()
    try:
        return set(json.loads(NAMES_FILE.read_text()).get("names", []))
    except (OSError, json.JSONDecodeError):
        return set()


def meta() -> dict:
    if not NAMES_FILE.exists():
        return {}
    try:
        data = json.loads(NAMES_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if k != "names"}


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------
def data_yaml() -> Path:
    return HOLDOUT_ROOT / "data.yaml"


def build(size: int = 1000, seed: int = 0, class_names: list[str] | None = None,
          nc: int = 13) -> Path:
    """Write pipeline/vehicles/holdout/ and record which images are in it."""
    from .build_fleet import class_names as read_class_names

    chosen = select(size, seed)
    if class_names is None:
        class_names, nc = read_class_names()

    images, labels = vehicles.image_index(), vehicles.label_index()
    (HOLDOUT_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (HOLDOUT_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    linked = 0
    for name in chosen:
        if name in images:
            vehicles.link(images[name], HOLDOUT_ROOT / "images" / "val" / name)
            linked += 1
        stem = name.rsplit(".", 1)[0]
        if stem in labels:
            vehicles.link(labels[stem], HOLDOUT_ROOT / "labels" / "val" / f"{stem}.txt")

    # `train` points at the same directory because Ultralytics wants the key to
    # exist. Nothing trains on this file -- only `val` is ever read, by val().
    data_yaml().write_text(
        f"path: {HOLDOUT_ROOT}\ntrain: images/val\nval: images/val\n"
        f"nc: {nc}\nnames:\n" + "".join(f"- {n}\n" for n in class_names)
    )
    NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NAMES_FILE.write_text(json.dumps(
        {"size": size, "seed": seed, "linked": linked, "names": chosen}, indent=1))
    return HOLDOUT_ROOT


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def checkpoints() -> list[Path]:
    """Global checkpoints in round order. ``global_last.pt`` is deliberately left
    out: it duplicates the highest round and would draw a false extra point."""
    ckpt_dir = paths.PROJECT / "checkpoints"
    rounds = sorted(ckpt_dir.glob("global_round_*.pt"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    return rounds


def evaluate(weights: Path, imgsz: int = 640, batch: int = 8, device: str = "0") -> dict:
    """Score one checkpoint on the holdout. Raises if it cannot -- never returns 0."""
    if not weights.exists():
        raise SystemExit(f"no such checkpoint: {weights}")
    if not data_yaml().exists():
        raise SystemExit("no holdout built yet; run `python -m pipeline.holdout --build`")

    from ultralytics import YOLO   # imported lazily: selection and tests need no torch

    result = YOLO(str(weights)).val(
        data=str(data_yaml()), imgsz=imgsz, batch=batch, device=device, workers=0,
        plots=False, verbose=False, project=str(paths.STATE / "holdout_runs"),
        name=weights.stem, exist_ok=True,
    )
    box = result.box
    return {"checkpoint": weights.name,
            "round": int(weights.stem.rsplit("_", 1)[-1]) if weights.stem[-1].isdigit() else None,
            "mAP50": float(box.map50), "mAP50-95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr)}


def evaluate_all(imgsz: int = 640, batch: int = 8, device: str = "0") -> list[dict]:
    """Score every global checkpoint and write the curve to .state/holdout_metrics.json."""
    ckpts = checkpoints()
    if not ckpts:
        raise SystemExit("no global_round_*.pt checkpoints; run the federate stage first")
    rows = [evaluate(c, imgsz=imgsz, batch=batch, device=device) for c in ckpts]
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps({"holdout": meta(), "rounds": rows}, indent=1))
    return rows


def curve() -> dict:
    """What the dashboard and the report read. ``{}`` before anything is scored."""
    try:
        return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="carve and materialise the holdout")
    ap.add_argument("--evaluate", action="store_true", help="score every global checkpoint on it")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    if not (args.build or args.evaluate):
        ap.error("pass --build, --evaluate, or both")

    if args.build:
        root = build(args.size, args.seed)
        info = meta()
        print(f"holdout: {info.get('linked')} of {info.get('size')} images materialised at {root}")
        print("no vehicle can train or self-evaluate on these: the fleet is assigned "
              "from the val pool with them removed")

    if args.evaluate:
        rows = evaluate_all(imgsz=args.imgsz, batch=args.batch, device=args.device)
        print(f"\nGlobal model on the shared holdout ({meta().get('size')} images "
              f"no vehicle saw):\n")
        for r in rows:
            print(f"  round {r['round']:<3} mAP50 {r['mAP50']:.4f}  "
                  f"mAP50-95 {r['mAP50-95']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}")
        if len(rows) > 1:
            delta = rows[-1]["mAP50"] - rows[0]["mAP50"]
            print(f"\n  {delta:+.4f} mAP50 across {len(rows)} rounds, measured on data "
                  f"no client trained on.")
        print(f"\nwritten: {METRICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
