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

import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, asdict, field
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

    def fingerprint(self) -> str:
        """Content hash of exactly which images this vehicle holds.

        Names, not bytes: the images are hardlinks onto a read-only cache, and what
        distinguishes one fleet from another is the assignment, not the pixels. 12
        hex characters is enough to tell two fleets apart in a table without making
        every row unreadable.
        """
        payload = "\n".join(["train", *self.train, "val", *self.val])
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_summary(self) -> dict:
        return {"vid": self.vid, "condition": self.condition,
                "n_train": len(self.train), "n_val": len(self.val),
                "fingerprint": self.fingerprint()}


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
def size_floor(per_vehicle: int) -> int:
    """The smallest shard a vehicle may be given.

    A shard below the batch size takes **no optimizer step at all** and logs a
    successful round anyway -- already in this project's catalogue of silent
    failures. So quantity skew is clamped rather than allowed to be as extreme as the
    draw suggests, and the clamp is one function because the fleet stage needs the
    same number to know whether a small shard is intended or stale.
    """
    return min(max(32, per_vehicle // 10), per_vehicle)


def skewed_sizes(n: int, per_vehicle: int, skew: float, rng) -> list[int]:
    """`n` shard sizes, lognormally spread, summing to exactly ``n * per_vehicle``.

    Lognormal because per-vehicle mileage is multiplicative: a car that drives twice
    as much as another is the realistic difference, not one that drives 40 images
    more. ``skew`` is the sigma of the underlying normal, so 0.5 is a mild spread and
    1.5 an order of magnitude between the busiest and quietest vehicle.

    The **total is preserved on purpose**. Skew must vary how the fleet's images are
    distributed, never how many there are -- otherwise a skewed arm and an unskewed
    one differ in image-visits too and neither explains the other, which is the
    multi-variable confound `compare.py` exists to flag.
    """
    total = n * per_vehicle
    floor = size_floor(per_vehicle)
    raw = [math.exp(rng.gauss(0.0, skew)) for _ in range(n)]
    scale = total / (sum(raw) or 1.0)
    sizes = [max(floor, int(round(r * scale))) for r in raw]

    # Rounding and the floor both push the sum off target. Repair it a unit at a time
    # from the largest shard down (or into the smallest), which keeps the shape of the
    # draw while making the total exact.
    # ponytail: O(drift x n) and drift is bounded by the fleet's size; fine for n < 100.
    drift = sum(sizes) - total
    while drift:
        step = -1 if drift > 0 else 1
        i = sizes.index(max(sizes)) if drift > 0 else sizes.index(min(sizes))
        if step < 0 and sizes[i] <= floor:
            break            # every shard is at the floor: n * floor > total, nothing to give
        sizes[i] += step
        drift += step
    return sizes


@dataclass
class Request:
    """One fleet's worth of parameters. What every partitioner is handed."""

    n_vehicles: int
    per_vehicle: int
    val_per_vehicle: int
    index: dict
    train_pool: set[str]
    val_pool: set[str]
    alpha: float = 0.5          # dirichlet only: smaller = more skewed
    #: Per-vehicle image budgets. Empty means every vehicle gets ``per_vehicle``,
    #: which is what every fleet built before quantity skew existed did.
    sizes: list[int] = field(default_factory=list)

    def budget(self, i: int) -> int:
        """How many train images vehicle ``i`` gets. Ask this, never ``per_vehicle``."""
        return self.sizes[i] if self.sizes else self.per_vehicle

    def val_budget(self, i: int) -> int:
        """Val scaled with train, so the ratio a vehicle self-evaluates at is constant.

        A vehicle holding a tenth of the fleet's images and the full val split would
        be evaluated more thoroughly than it was trained, which is not what a smaller
        vehicle means.
        """
        if not self.sizes:
            return self.val_per_vehicle
        share = self.sizes[i] / max(1, self.per_vehicle)
        return max(5, int(round(self.val_per_vehicle * share)))


#: name -> partitioner. Registering is the only step needed to make a strategy
#: reachable from the CLI, the dashboard and the fleet check -- all three read
#: PARTITIONS, which is this dict's keys.
PARTITIONERS: dict[str, callable] = {}


def partitioner(name: str):
    """Register a partition strategy: ``fn(req: Request, rng) -> list[Vehicle]``.

    A partitioner must be deterministic given ``rng`` and must hand out disjoint
    slices -- overlap would let one image train two vehicles in a round, which is
    not what a fleet does and would quietly flatter the aggregate.
    """
    def register(fn):
        PARTITIONERS[name] = fn
        return fn
    return register


def _picker(req: Request, rng, used: set[str]):
    """Take `want` unused images matching a predicate, topping up if short."""
    def pick(pool: set[str], want: int, matches) -> list[str]:
        on = [n for n in pool if n not in used and matches(req.index.get(n, {}))]
        rng.shuffle(on)
        chosen = on[:want]
        if len(chosen) < want:
            # Not enough images match this condition -- top up with anything unused
            # so the vehicle still trains, and let the caller see the shortfall in
            # the summary rather than silently getting a tiny shard.
            rest = [n for n in pool if n not in used and n not in set(chosen)]
            rng.shuffle(rest)
            chosen += rest[: want - len(chosen)]
        used.update(chosen)
        return sorted(chosen)
    return pick


def _by_profile(req: Request, rng, profile_for) -> list[Vehicle]:
    """Shared body of every predicate-based partitioner."""
    used: set[str] = set()
    out: list[Vehicle] = []
    pick = _picker(req, rng, used)
    for i in range(req.n_vehicles):
        label, matches = profile_for(i)
        out.append(Vehicle(i + 1, label,
                           pick(req.train_pool, req.budget(i), matches),
                           pick(req.val_pool, req.val_budget(i), matches)))
    return out


@partitioner("condition")
def _assign_condition(req: Request, rng) -> list[Vehicle]:
    """Each vehicle biased toward one driving condition. The interesting case:
    divergence between vehicles is the expected result."""
    return _by_profile(req, rng, lambda i: PROFILES[i % len(PROFILES)])


ANY = (lambda a: True)


@partitioner("random")
def _assign_random(req: Request, rng) -> list[Vehicle]:
    """Uniform random slices. IID, and the control: these curves *should* converge,
    and if they do not, something other than the data is going on."""
    return _by_profile(req, rng, lambda i: ("random mix", ANY))


@partitioner("mixed")
def _assign_mixed(req: Request, rng) -> list[Vehicle]:
    """Alternating, so one run shows both behaviours side by side."""
    return _by_profile(req, rng,
                       lambda i: ("random mix", ANY) if i % 2 else PROFILES[i % len(PROFILES)])


# ---------------------------------------------------------------- dirichlet
def _group_of(attrs: dict) -> str:
    """Which condition an image belongs to, by the same predicates PROFILES uses.

    One definition of "a driving condition" for the whole package: if a profile's
    predicate changes, the Dirichlet groups change with it.
    """
    for label, matches in PROFILES:
        if matches(attrs):
            return label
    return "other"


def _groups(pool: set[str], index: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in pool:
        out.setdefault(_group_of(index.get(name, {})), []).append(name)
    for names in out.values():
        names.sort()                      # deterministic before any shuffle
    return out


def _dirichlet_draw(k: int, alpha: float, rng) -> list[float]:
    """A Dir(alpha) sample over k categories, from stdlib gammas.

    Dir(alpha) is k independent Gamma(alpha, 1) draws normalised -- no numpy needed,
    and one fewer import that has to exist inside a Ray worker.
    """
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(xs) or 1.0
    return [x / total for x in xs]


@partitioner("dirichlet")
def _assign_dirichlet(req: Request, rng) -> list[Vehicle]:
    """Each vehicle draws its own mixture over conditions from Dir(alpha).

    alpha -> 0 concentrates a vehicle on a single condition; alpha -> infinity gives
    every vehicle the same mixture, which is IID. This is the knob FL papers report
    against, so a run here can be compared with published ones.

    The per-client-mixture variant is used rather than the per-group split, so shard
    sizes stay equal: quantity skew and distribution skew are different experiments
    and mixing them makes neither readable. See docs/prompts/2026-08-06-dirichlet-
    partition.md for the rejected variant.
    """
    if req.alpha <= 0:
        raise ValueError(f"dirichlet alpha must be > 0, got {req.alpha}")

    train_groups = _groups(req.train_pool, req.index)
    val_groups = _groups(req.val_pool, req.index)
    keys = sorted(train_groups)
    used: set[str] = set()
    out: list[Vehicle] = []

    def take(groups: dict[str, list[str]], shares: list[float], want: int) -> list[str]:
        chosen: list[str] = []
        for key, share in zip(keys, shares):
            # Clamped to what is left: rounding each group's share independently can
            # sum to more than the budget, which would hand one vehicle a bigger
            # shard than another and turn the FedAvg weights into a second variable.
            quota = min(int(round(share * want)), want - len(chosen))
            available = [n for n in groups.get(key, []) if n not in used]
            rng.shuffle(available)
            picked = available[:quota]
            used.update(picked)
            chosen += picked
        if len(chosen) < want:               # rounding, or a group ran dry
            rest = [n for names in groups.values() for n in names if n not in used]
            rng.shuffle(rest)
            top_up = rest[: want - len(chosen)]
            used.update(top_up)
            chosen += top_up
        return sorted(chosen)

    for i in range(req.n_vehicles):
        shares = _dirichlet_draw(len(keys), req.alpha, rng)
        train = take(train_groups, shares, req.budget(i))
        val = take(val_groups, shares, req.val_budget(i))

        # Label by what the vehicle actually got, not by what was drawn: a group
        # that ran dry is exactly the case a label must not hide.
        got: dict[str, int] = {}
        for name in train:
            key = _group_of(req.index.get(name, {}))
            got[key] = got.get(key, 0) + 1
        top = max(got, key=got.get) if got else "empty"
        pct = round(100 * got.get(top, 0) / max(1, len(train)))
        out.append(Vehicle(i + 1, f"dirichlet a={req.alpha:g} - {top} {pct}%", train, val))
    return out


PARTITIONS = tuple(PARTITIONERS)


def assign(n_vehicles: int, per_vehicle: int, val_per_vehicle: int = 0,
           seed: int = 0, index: dict | None = None,
           train_pool: set[str] | None = None, val_pool: set[str] | None = None,
           partition: str = "condition", alpha: float = 0.5,
           exclude: set[str] | None = None, size_skew: float = 0.0) -> list[Vehicle]:
    """Give each vehicle a disjoint slice, by whichever registered strategy is named.

    ``exclude`` is removed from both pools before anything is picked -- it is how the
    shared holdout stays held out. Subtracting here rather than inside each
    partitioner means a new strategy cannot forget to do it.

    ``size_skew`` is orthogonal to ``partition``: it decides how *much* each vehicle
    holds, where the partition decides *what*. Drawn here rather than inside a
    partitioner so every strategy gets it for free and none can forget it. At the
    default 0 not a single number is drawn, so every fleet built before skew existed
    still reproduces from its seed.

    ``train_pool``/``val_pool`` are injectable so this is testable without a populated
    repo; they default to the images actually present in my-project's shards.
    """
    if partition not in PARTITIONERS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if size_skew < 0:
        raise ValueError(f"size_skew must be >= 0, got {size_skew}")

    rng = random.Random(seed)
    blocked = set(exclude or ())
    req = Request(
        n_vehicles=n_vehicles,
        per_vehicle=per_vehicle,
        val_per_vehicle=val_per_vehicle or max(20, per_vehicle // 5),
        index=index if index is not None else build_attribute_index(),
        train_pool=(_available(Path("train")) if train_pool is None else train_pool) - blocked,
        val_pool=(_available(Path("val")) if val_pool is None else val_pool) - blocked,
        alpha=alpha,
        sizes=skewed_sizes(n_vehicles, per_vehicle, size_skew, rng) if size_skew else [],
    )
    return PARTITIONERS[partition](req, rng)


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------
def label_index() -> dict[str, Path]:
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


def image_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for batch in sorted(paths.PROJECT.glob("batch/batch_*")):
        for split in ("train", "val"):
            d = batch / "images" / split
            if d.is_dir():
                with os.scandir(d) as it:
                    for e in it:
                        out.setdefault(e.name, Path(e.path))
    return out


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hardlink: a fleet costs no extra disk
    except OSError:
        shutil.copy2(src, dst)


def materialise(vehicles: list[Vehicle], class_names: list[str], nc: int = 13,
                meta: dict | None = None) -> Path:
    """Write pipeline/vehicles/batch/batch_<vid>/ for each vehicle. Returns the root.

    ``meta`` -- partition, alpha, seed, images per vehicle -- is written beside the
    summaries as ``fleet.meta.json``. Before it existed, the fleet stage decided
    whether a rebuild was needed by checking whether every label read "random mix",
    which cannot tell a condition fleet from a mixed one.
    """
    images, labels = image_index(), label_index()
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
                    link(images[name], bd / "images" / split / name)
                stem = name.rsplit(".", 1)[0]
                if stem in labels:
                    link(labels[stem], bd / "labels" / split / f"{stem}.txt")
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
    # One digest over every vehicle's assignment: two runs carrying the same fleet
    # fingerprint trained on exactly the same images, which is what makes a seed
    # repeat a repeat and a strategy comparison a comparison.
    digest = hashlib.sha256(
        "|".join(f"{v.vid}:{v.fingerprint()}" for v in vehicles).encode()).hexdigest()[:12]
    (root.parent / "fleet.meta.json").write_text(
        json.dumps({**(meta or {}), "fingerprint": digest}, indent=1))
    return paths.VEHICLE_ROOT


def load_fleet() -> list[dict]:
    f = paths.VEHICLE_ROOT / "fleet.json"
    return json.loads(f.read_text()) if f.exists() else []


def load_fleet_meta() -> dict:
    """How the fleet on disk was built. ``{}`` for a fleet built before manifests."""
    f = paths.VEHICLE_ROOT / "fleet.meta.json"
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------
# What a shard actually contains
# --------------------------------------------------------------------------
_ATTR_MEMO: dict | None = None


def cached_attributes() -> dict:
    """The attribute index if it has already been built, else ``{}``.

    Deliberately never builds it: this is called from a request handler, and
    streaming the 1.45 GB label JSON there would hang the page for minutes with no
    way to see why. An empty index degrades to "unknown", which is honest.
    """
    global _ATTR_MEMO
    if _ATTR_MEMO is None:
        _ATTR_MEMO = json.loads(ATTR_CACHE.read_text()) if ATTR_CACHE.exists() else {}
    return _ATTR_MEMO


def composition(vid: int, samples: int = 8, index: dict | None = None) -> dict:
    """Attribute breakdown of one vehicle's own shard, plus a few image names.

    "Vehicle 3 is the rain / fog one" is a claim about the data, and until now the
    only evidence for it was the label the assignment printed. This counts what the
    shard holds, so a condition that silently topped up with random images shows as
    the mixture it is.
    """
    bd = paths.VEHICLE_BATCHES / f"batch_{vid}"
    if not bd.is_dir():
        return {"vid": vid, "error": "no shard for this vehicle"}

    listing = bd / "train.txt"
    names = [n.strip() for n in listing.read_text().splitlines() if n.strip()] if listing.exists() else []
    index = cached_attributes() if index is None else index

    counts: dict[str, dict[str, int]] = {"weather": {}, "scene": {}, "timeofday": {}}
    for name in names:
        attrs = index.get(name) or {}
        for key, bucket in counts.items():
            value = attrs.get(key) or "unknown"
            bucket[value] = bucket.get(value, 0) + 1

    train_dir = bd / "images" / "train"
    present: list[str] = []
    for name in names:
        if len(present) >= samples:
            break
        if (train_dir / name).is_file():
            present.append(name)

    return {
        "vid": vid,
        "n_train": len(names),
        "n_val": paths.count_files(bd / "images" / "val"),
        "indexed": bool(index),
        "counts": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in counts.items()},
        "samples": present,
    }


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

    skewed = assign(4, 50, size_skew=1.0, **kw)
    sizes = [v.n_train for v in skewed]
    assert len(set(sizes)) > 1, f"skew produced equal shards: {sizes}"
    assert sum(sizes) == 200, f"skew changed the fleet's budget: {sizes}"
    assert min(sizes) >= size_floor(50), f"a shard below the batch size: {sizes}"
    print("vehicles self-check OK:", [v.to_summary() for v in vs], "skewed sizes:", sizes)


if __name__ == "__main__":
    demo()
