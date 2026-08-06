"""The centralised baseline: what one model reaches on the pooled data.

Federated numbers have no scale without it. 0.455 mAP50 is either near the ceiling
for this budget or half of it, and until a centralised model is trained on the union
of the same images, for the same number of image-visits, and scored on the same
holdout, nothing here can be called a success or a failure.

The budget is matched by construction. A federated run does
``rounds x local_epochs`` epochs on each of ``n_vehicles`` shards, so the total
image-visits equal ``rounds x local_epochs`` epochs over the pooled set. That is the
default for ``--epochs``, and it is why the comparison is fair rather than
flattering.

    python -m pipeline.baseline --epochs 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import holdout, paths, vehicles

POOLED_ROOT = paths.VEHICLE_ROOT / "pooled"          # gitignored with the rest
RESULT_FILE = paths.STATE / "baseline.json"


def pooled_names() -> list[str]:
    """Every image the fleet trains on, deduplicated. The centralised model's data."""
    seen: dict[str, None] = {}
    for listing in sorted(paths.VEHICLE_BATCHES.glob("batch_*/train.txt")):
        for line in listing.read_text().splitlines():
            name = line.strip()
            if name:
                seen[name] = None
    return list(seen)


def build(class_names: list[str] | None = None, nc: int = 13) -> Path:
    """Materialise the pooled training set, validated against the shared holdout."""
    from .build_fleet import class_names as read_class_names

    names = pooled_names()
    if not names:
        raise SystemExit("no fleet on disk to pool; run the fleet stage first")
    if not holdout.data_yaml().exists():
        raise SystemExit("no holdout to validate against; run "
                         "`python -m pipeline.holdout --build` first")
    if class_names is None:
        class_names, nc = read_class_names()

    images, labels = vehicles.image_index(), vehicles.label_index()
    (POOLED_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (POOLED_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    for name in names:
        if name in images:
            vehicles.link(images[name], POOLED_ROOT / "images" / "train" / name)
        stem = name.rsplit(".", 1)[0]
        if stem in labels:
            vehicles.link(labels[stem], POOLED_ROOT / "labels" / "train" / f"{stem}.txt")

    # val is the shared holdout, by absolute path: the centralised model and the
    # federated one must be scored on exactly the same images or the gap means
    # nothing.
    (POOLED_ROOT / "data.yaml").write_text(
        f"path: {POOLED_ROOT}\ntrain: images/train\n"
        f"val: {holdout.HOLDOUT_ROOT / 'images' / 'val'}\n"
        f"nc: {nc}\nnames:\n" + "".join(f"- {n}\n" for n in class_names)
    )
    return POOLED_ROOT


def train(epochs: int, imgsz: int = 640, batch: int = 16, device: str = "0",
          model: str = "models/yolov8s-13.yaml", pretrained: str = "models/yolov8s.pt") -> dict:
    """Train one model on the pooled data and score it on the holdout.

    Same architecture and same pretrained weights the clients use, so the only
    thing that differs from the federated run is that the data was pooled.
    """
    from ultralytics import YOLO

    yolo = YOLO(str(paths.PROJECT / model))
    yolo.train(
        data=str(POOLED_ROOT / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, workers=0, plots=False, verbose=True, pretrained=str(paths.PROJECT / pretrained),
        project=str(paths.STATE / "baseline_runs"), name="centralised", exist_ok=True,
    )
    result = yolo.val(data=str(holdout.data_yaml()), imgsz=imgsz, batch=batch, device=device,
                      workers=0, plots=False, verbose=False,
                      project=str(paths.STATE / "baseline_runs"), name="centralised_val",
                      exist_ok=True)
    box = result.box
    return {"epochs": epochs, "images": len(pooled_names()), "imgsz": imgsz,
            "mAP50": float(box.map50), "mAP50-95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr)}


def result() -> dict:
    try:
        return json.loads(RESULT_FILE.read_text()) if RESULT_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def gap() -> dict:
    """Federated best against the centralised ceiling, on the same holdout."""
    central = result()
    rounds = holdout.curve().get("rounds") or []
    if not central or not rounds:
        return {}
    best = max(r["mAP50"] for r in rounds)
    return {"federated_mAP50": best, "centralised_mAP50": central["mAP50"],
            "gap": central["mAP50"] - best,
            "retained": best / central["mAP50"] if central["mAP50"] else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = rounds x local_epochs, which matches the federated budget")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--local-epochs", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    epochs = args.epochs or args.rounds * args.local_epochs
    root = build()
    n = len(pooled_names())
    print(f"pooled: {n} images at {root}, {epochs} epochs "
          f"(matching {args.rounds} rounds x {args.local_epochs} local epochs)")

    row = train(epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(row, indent=1))

    print(f"\ncentralised on the shared holdout: mAP50 {row['mAP50']:.4f}  "
          f"mAP50-95 {row['mAP50-95']:.4f}")
    g = gap()
    if g:
        print(f"federated best: {g['federated_mAP50']:.4f}  "
              f"gap: {g['gap']:+.4f}  federation retains "
              f"{100 * g['retained']:.1f}% of the centralised ceiling")
    else:
        print("no federated holdout curve yet; run `python -m pipeline.holdout --evaluate`")
    print(f"\nwritten: {RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
