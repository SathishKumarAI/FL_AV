"""Simulated vehicles: each one a Flower client seeing a different slice of the world.

Federated learning is only interesting when clients see *different* data. Handing
every vehicle a random shard makes their curves converge to the same shape and the
fleet view says nothing. So each vehicle is biased toward a driving condition drawn
from BDD100K's own per-image attributes (weather / scene / timeofday).

Shards are written under ``pipeline/vehicles/batch/`` and reached by pointing the
existing ``FL_AV_DATA_ROOT`` env var at ``pipeline/vehicles``. Nothing is created
inside ``my-project`` and no source there changes.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

from . import paths

ATTR_CACHE = paths.STATE / "attributes.json"

# Each vehicle is a (label, predicate) over BDD's attribute dict. Order matters:
# vehicle 1 gets the first profile, and profiles repeat if more vehicles are asked
# for than exist here.
PROFILES: list[tuple[str, callable]] = [
    ("daytime city", lambda a: a.get("timeofday") == "daytime" and a.get("scene") == "city street"),
    ("night", lambda a: a.get("timeofday") == "night"),
    ("rain / fog", lambda a: a.get("weather") in {"rainy", "foggy"}),
    ("highway", lambda a: a.get("scene") == "highway"),
    ("dawn / dusk", lambda a: a.get("timeofday") == "dawn/dusk"),
    ("overcast residential", lambda a: a.get("weather") == "overcast" and a.get("scene") == "residential"),
    ("snow", lambda a: a.get("weather") == "snowy"),
    ("parking / tunnel", lambda a: a.get("scene") in {"parking lot", "tunnel"}),
]


@dataclass
class Vehicle:
    vid: int
    condition: str
    train: list[str]
    val: list[str]

    @property
    def n_train(self) -> int:
        return len(self.train)

    def to_summary(self) -> dict:
        return {"vid": self.vid, "condition": self.condition,
                "n_train": len(self.train), "n_val": len(self.val)}


# --------------------------------------------------------------------------
# Attribute index
# --------------------------------------------------------------------------
def build_attribute_index(force: bool = False) -> dict[str, dict]:
    """name -> {weather, scene, timeofday}, streamed and cached.

    The train label file is 1.45 GB; ``json.loads`` on it would need several GB of
    RAM to produce objects we throw away immediately. ijson streams it, we keep only
    the three attribute strings per image, and the result is cached so this cost is
    paid once.
    """
    if ATTR_CACHE.exists() and not force:
        return json.loads(ATTR_CACHE.read_text())

    import ijson  # imported lazily: only the first index build needs it

    index: dict[str, dict] = {}
    for jf in paths.find_label_jsons():
        with open(jf, "rb") as fh:
            for rec in ijson.items(fh, "item"):
                attrs = rec.get("attributes") or {}
                index[rec["name"]] = {
                    "weather": attrs.get("weather"),
                    "scene": attrs.get("scene"),
                    "timeofday": attrs.get("timeofday"),
                }
    ATTR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ATTR_CACHE.write_text(json.dumps(index, separators=(",", ":")))
    return index


def _available(split_dir: Path) -> set[str]:
    """Image basenames actually populated for a split, across all source shards."""
    names: set[str] = set()
    for batch in sorted(paths.PROJECT.glob("batch/batch_*")):
        d = batch / "images" / split_dir.name
        if d.is_dir():
            with os.scandir(d) as it:
                names.update(e.name for e in it if e.is_file())
    return names


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------
def assign(n_vehicles: int, per_vehicle: int, val_per_vehicle: int = 0,
           seed: int = 0, index: dict | None = None,
           train_pool: set[str] | None = None, val_pool: set[str] | None = None) -> list[Vehicle]:
    """Give each vehicle a condition-biased, disjoint slice.

    Disjoint matters: overlapping shards would let the same image train two vehicles
    in one round, which is not what a fleet does and would quietly flatter the
    aggregate.

    ``train_pool``/``val_pool`` are injectable so this is testable without a populated
    repo; they default to the images actually present in my-project's shards.
    """
    index = index if index is not None else build_attribute_index()
    rng = random.Random(seed)

    train_pool = _available(Path("train")) if train_pool is None else train_pool
    val_pool = _available(Path("val")) if val_pool is None else val_pool
    val_per_vehicle = val_per_vehicle or max(20, per_vehicle // 5)

    used: set[str] = set()
    vehicles: list[Vehicle] = []
    for i in range(n_vehicles):
        label, matches = PROFILES[i % len(PROFILES)]

        def pick(pool: set[str], want: int) -> list[str]:
            on = [n for n in pool if n not in used and matches(index.get(n, {}))]
            rng.shuffle(on)
            chosen = on[:want]
            if len(chosen) < want:
                # Not enough images match this condition -- top up with anything
                # unused so the vehicle still trains, and let the caller see the
                # shortfall in the summary rather than silently getting a tiny shard.
                rest = [n for n in pool if n not in used and n not in set(chosen)]
                rng.shuffle(rest)
                chosen += rest[: want - len(chosen)]
            used.update(chosen)
            return sorted(chosen)

        vehicles.append(Vehicle(i + 1, label, pick(train_pool, per_vehicle), pick(val_pool, val_per_vehicle)))
    return vehicles


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------
def _label_index() -> dict[str, Path]:
    """basename(without .txt) -> label file, across my-project's shards (read-only)."""
    out: dict[str, Path] = {}
    for batch in sorted(paths.PROJECT.glob("batch/batch_*")):
        for split in ("train", "val"):
            d = batch / "labels" / split
            if d.is_dir():
                with os.scandir(d) as it:
                    for e in it:
                        if e.name.endswith(".txt"):
                            out.setdefault(e.name[:-4], Path(e.path))
    return out


def _image_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for batch in sorted(paths.PROJECT.glob("batch/batch_*")):
        for split in ("train", "val"):
            d = batch / "images" / split
            if d.is_dir():
                with os.scandir(d) as it:
                    for e in it:
                        out.setdefault(e.name, Path(e.path))
    return out


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hardlink: a fleet costs no extra disk
    except OSError:
        shutil.copy2(src, dst)


def materialise(vehicles: list[Vehicle], class_names: list[str], nc: int = 13) -> Path:
    """Write pipeline/vehicles/batch/batch_<vid>/ for each vehicle. Returns the root."""
    images, labels = _image_index(), _label_index()
    root = paths.VEHICLE_BATCHES
    if root.exists():
        shutil.rmtree(root)        # a fleet is defined per run; stale shards would lie

    for v in vehicles:
        bd = root / f"batch_{v.vid}"
        for split, names in (("train", v.train), ("val", v.val)):
            (bd / "images" / split).mkdir(parents=True, exist_ok=True)
            (bd / "labels" / split).mkdir(parents=True, exist_ok=True)
            for name in names:
                if name in images:
                    _link(images[name], bd / "images" / split / name)
                stem = name.rsplit(".", 1)[0]
                if stem in labels:
                    _link(labels[stem], bd / "labels" / split / f"{stem}.txt")
            (bd / f"{split}.txt").write_text("".join(f"{n}\n" for n in names))
        (bd / "test.txt").write_text("")
        # `path` is a placeholder; the client rewrites it via materialize_data_yaml().
        (bd / "data.yaml").write_text(
            f"path: {bd}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"nc: {nc}\nnames:\n" + "".join(f"- {n}\n" for n in class_names)
        )
    # Summary only: the full name lists live in each shard's train.txt/val.txt, and
    # embedding them here made fleet.json megabytes wide while still omitting n_val,
    # which is why the report printed "val | ?" for every vehicle.
    (root.parent / "fleet.json").write_text(
        json.dumps([v.to_summary() for v in vehicles], indent=1)
    )
    return paths.VEHICLE_ROOT


def load_fleet() -> list[dict]:
    f = paths.VEHICLE_ROOT / "fleet.json"
    return json.loads(f.read_text()) if f.exists() else []


def demo() -> None:
    """Self-check on a synthetic index: bias applied, slices disjoint, deterministic."""
    index = {}
    for i in range(400):
        index[f"img{i}.jpg"] = {
            "timeofday": "night" if i % 2 else "daytime",
            "scene": "city street", "weather": "clear",
        }
    pool = set(index)
    kw = dict(index=index, train_pool=pool, val_pool=pool, val_per_vehicle=10, seed=1)

    vs = assign(2, 50, **kw)
    assert [v.condition for v in vs] == ["daytime city", "night"], vs
    assert all(index[n]["timeofday"] == "daytime" for n in vs[0].train), "bias not applied"
    assert all(index[n]["timeofday"] == "night" for n in vs[1].train), "bias not applied"
    assert not (set(vs[0].train) & set(vs[1].train)), "slices overlap"
    assert [v.train for v in assign(2, 50, **kw)] == [v.train for v in vs], "not deterministic"
    print("vehicles self-check OK:", [v.to_summary() for v in vs])


if __name__ == "__main__":
    demo()
