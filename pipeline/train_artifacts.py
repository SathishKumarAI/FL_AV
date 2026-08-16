"""What YOLO actually consumed, as pictures rather than as counts.

`dataset_stats` answers *how much* of each class a shard holds. It cannot answer the
question a wrong run always turns out to hinge on: **is the model looking at what we
think it is looking at?** Mosaic augmentation, letterboxing, a label file whose class
ids are off by one, a shard whose "night" images are daytime — none of that is visible
in a histogram, and all of it is visible in one glance at a batch.

Ultralytics already draws every one of those pictures. Each client's `train()` writes
them beside its checkpoints and nothing has ever looked at them:

    train_batch{0,1,2}.jpg      the real batches, post-augmentation, boxes drawn
    labels.jpg                  class counts + box size/position distributions
    val_batch{n}_labels.jpg     ground truth on the val split
    val_batch{n}_pred.jpg       this model's predictions on the same images
    confusion_matrix*.png       which classes it confuses for which
    Box{P,R,F1,PR}_curve.png    precision/recall against confidence
    results.png                 the loss and metric curves for this vehicle

So this module writes nothing and renders nothing. It locates those files per vehicle
and hands them to the dashboard. Assemble before building, rule 4.

Two caveats the dashboard repeats to the reader, because a picture is persuasive in a
way a number is not:

- the client passes ``exist_ok=True``, so a vehicle's directory holds only its **last**
  round. These are not a history.
- ``val_batch*_pred.jpg`` is that client's local model on its *own* val split, not the
  aggregate on the holdout. It is the flattering number made visual.

    python -m pipeline.train_artifacts        # what exists, per vehicle
"""
from __future__ import annotations

import argparse

from . import paths, vehicles

#: filename -> (group, what it proves). Order is the display order. Anything the
#: trainer writes that is not in here is not served: an allowlist, so a future
#: ultralytics version cannot expose a new file by URL without a decision here.
KINDS: dict[str, tuple[str, str]] = {
    "train_batch0.jpg": ("consumed", "a real training batch, after mosaic and augmentation"),
    "train_batch1.jpg": ("consumed", "a real training batch, after mosaic and augmentation"),
    "train_batch2.jpg": ("consumed", "a real training batch, after mosaic and augmentation"),
    "labels.jpg": ("consumed", "class counts, and where the boxes sit in the frame"),
    "val_batch0_labels.jpg": ("truth", "ground truth on this vehicle's val split"),
    "val_batch0_pred.jpg": ("pred", "what this vehicle's local model predicted there"),
    "val_batch1_labels.jpg": ("truth", "ground truth on this vehicle's val split"),
    "val_batch1_pred.jpg": ("pred", "what this vehicle's local model predicted there"),
    "val_batch2_labels.jpg": ("truth", "ground truth on this vehicle's val split"),
    "val_batch2_pred.jpg": ("pred", "what this vehicle's local model predicted there"),
    "confusion_matrix_normalized.png": ("quality", "which classes it mistakes for which"),
    "BoxPR_curve.png": ("quality", "precision against recall, per class"),
    "BoxF1_curve.png": ("quality", "F1 against confidence — where to set the threshold"),
    "results.png": ("quality", "this vehicle's own loss and metric curves"),
}

#: The pairs the dashboard shows side by side. Truth on the left, prediction on the
#: right, same images — the only honest way to look at a detector.
PAIRS = [(f"val_batch{n}_labels.jpg", f"val_batch{n}_pred.jpg") for n in range(3)]


def run_dir(vid: int):
    """The ultralytics run directory for one vehicle, or None.

    Globbed rather than constructed. The client passes ``project="runs/fl"`` relative
    to its own CWD, and ultralytics resolves that against its ``runs_dir`` setting —
    which is why the path on this machine is `runs/detect/runs/fl/batch1` and would be
    `runs/fl/batch1` on a machine whose settings were never touched. Constructing it
    would work here and break for anyone reproducing the project.
    """
    found = [d for d in paths.PROJECT.glob(f"runs/**/batch{int(vid)}")
             if d.is_dir() and (d / "args.yaml").exists()]
    return max(found, key=lambda d: d.stat().st_mtime, default=None)


def artifact(vid: int, name: str):
    """``name`` inside that vehicle's run directory, if it is one we serve."""
    if name not in KINDS:
        return None
    root = run_dir(vid)
    if root is None:
        return None
    target = root / name
    return target if target.is_file() else None


def for_vehicle(vid: int) -> dict:
    root = run_dir(vid)
    if root is None:
        return {"vid": vid, "dir": None, "files": []}
    files = []
    for name, (group, caption) in KINDS.items():
        f = root / name
        if f.is_file():
            files.append({"name": name, "group": group, "caption": caption,
                          "mtime": int(f.stat().st_mtime)})
    return {"vid": vid, "dir": str(root), "files": files}


def listing() -> dict:
    """Every vehicle in the current fleet, with whatever its last round left behind.

    Keyed off the fleet rather than off the directories on disk, so a vehicle that has
    never trained shows up as empty instead of silently missing — the distinction
    between "no artifacts" and "no such vehicle" is the one worth seeing.
    """
    fleet = vehicles.load_fleet()
    return {
        "pairs": [list(p) for p in PAIRS],
        "vehicles": [{**for_vehicle(int(v["vid"])),
                      "condition": v.get("condition")} for v in fleet if v.get("vid") is not None],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    d = listing()
    if not d["vehicles"]:
        print("no fleet on disk — run the fleet stage first")
        return 1
    for v in d["vehicles"]:
        where = v["dir"] or "never trained"
        print(f"v{v['vid']:<3} {str(v.get('condition') or '')[:26]:<28} "
              f"{len(v['files']):>2} artifacts  {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
