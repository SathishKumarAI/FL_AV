"""The pipeline stages: what each one runs, when it can be skipped, and what it costs.

Every stage is a subprocess with ``cwd=my-project``. That is not incidental — flwr's
detached SuperLink caches the CWD of whichever run started it, so launching from
anywhere else makes every relative path in the project resolve somewhere wrong.

A stage that fails halts the chain. This project's history is a catalogue of silent
no-ops (weights returned untrained, every client handed the same shard, a round with
no optimizer step), so nothing here is allowed to continue past a failure.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import paths, vehicles

PY = sys.executable


@dataclass
class Config:
    """Everything a run is parameterised by. Also the 'inputs' half of the report."""

    profile: str = "demo"        # demo | full
    n_vehicles: int = 6
    rounds: int = 2
    local_epochs: int = 1
    seed: int = 0

    @property
    def per_vehicle(self) -> int:
        return 300 if self.profile == "demo" else 6308

    @property
    def imgsz(self) -> int:
        return 320 if self.profile == "demo" else 640

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(per_vehicle=self.per_vehicle, imgsz=self.imgsz)
        return d


@dataclass
class Check:
    satisfied: bool
    detail: str


@dataclass
class Stage:
    name: str
    title: str
    gated: bool                      # costs real time or GPU -> needs confirmation
    check: callable                  # (Config) -> Check
    command: callable                # (Config) -> list[str] | None
    cwd: Path = paths.PROJECT
    data_root: Path | None = None    # overrides FL_AV_DATA_ROOT for this stage
    est: str = ""


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def _check_env(_: Config) -> Check:
    # Deliberately never "satisfied": it is a two-second probe and the answer
    # (right torch build? GPU visible?) is the single most useful line of output.
    return Check(False, "always runs; probes torch + CUDA capability")


def _check_dataset(_: Config) -> Check:
    pool = paths.find_pool()
    if pool is None:
        return Check(False, "BDD100K pool not found in the kagglehub cache")
    n_tr = paths.count_files(pool / "train", {".jpg"})
    n_va = paths.count_files(pool / "val", {".jpg"})
    # The Kaggle mirror nests train into trainA/trainB/..., so a flat count reads low
    # even when the download is complete. Val is flat and is the reliable signal.
    if n_va >= 10000:
        return Check(True, f"pool present ({n_va} val, {n_tr} loose train + nested)")
    return Check(False, f"pool incomplete ({n_va} val images)")


def _check_populate(_: Config) -> Check:
    missing = []
    for bid in paths.BATCH_IDS:
        batch = paths.PROJECT / "batch" / f"batch_{bid}"
        listing = batch / "train.txt"
        if not listing.exists():
            missing.append(f"batch_{bid}: no train.txt")
            continue
        want = sum(1 for l in listing.read_text().splitlines() if l.strip())
        have = paths.count_files(batch / "images" / "train", {".jpg"})
        if have < want:
            missing.append(f"batch_{bid}: {have}/{want}")
    if missing:
        return Check(False, "shards incomplete -- " + ", ".join(missing[:3]))
    return Check(True, "all 10 shards match their split lists")


def _check_fleet(cfg: Config) -> Check:
    fleet = vehicles.load_fleet()
    if len(fleet) != cfg.n_vehicles:
        return Check(False, f"fleet has {len(fleet)} vehicles, want {cfg.n_vehicles}")
    short = [v for v in fleet if v.get("n_train", 0) < cfg.per_vehicle]
    if short:
        return Check(False, f"{len(short)} vehicle(s) below {cfg.per_vehicle} images")
    return Check(True, f"{len(fleet)} vehicles: " + ", ".join(v["condition"] for v in fleet))


def _check_sanity(_: Config) -> Check:
    marker = paths.STATE / "sanity.ok"
    return Check(marker.exists(), "passed previously" if marker.exists() else "not yet run")


def _check_federate(_: Config) -> Check:
    return Check(False, "always runs; this is the point of the pipeline")


def _check_verify(_: Config) -> Check:
    return Check(False, "always runs; asserts the four pass criteria")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
_ENV_PROBE = (
    "import torch;"
    "cap=torch.cuda.get_device_capability() if torch.cuda.is_available() else None;"
    "print('torch',torch.__version__,'cuda',torch.version.cuda,'capability',cap);"
    "print('device',torch.cuda.get_device_name(0) if cap else 'CPU ONLY');"
    "assert cap is not None,'no CUDA device visible -- the run would silently use CPU'"
)


def _cmd_env(_: Config) -> list[str]:
    return [PY, "-c", _ENV_PROBE]


def _cmd_dataset(_: Config) -> list[str]:
    return [PY, "-c",
            "import kagglehub;"
            f"print('path:', kagglehub.dataset_download('{paths.KAGGLE_DATASET}'))"]


def _cmd_populate(_: Config) -> list[str] | None:
    pool = paths.find_pool()
    if pool is None:
        raise RuntimeError("cannot populate: dataset pool not found; run the dataset stage")
    return [PY, "scripts/populate_images.py", "--pool", str(pool)]


def _cmd_fleet(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.build_fleet",
            "--vehicles", str(cfg.n_vehicles),
            "--per-vehicle", str(cfg.per_vehicle),
            "--seed", str(cfg.seed)]


def _cmd_sanity(cfg: Config) -> list[str]:
    return [PY, "-m", "ultralytics.cfg", "detect", "train",
            "data=batch/batch_1/data.runtime.yaml",
            "model=models/yolov8s-13.yaml", "pretrained=models/yolov8s.pt",
            "epochs=1", f"imgsz={cfg.imgsz}", "batch=4", "device=0", "workers=2",
            "plots=False", "project=runs/pipeline", "name=sanity", "exist_ok=True"]


def _cmd_federate(cfg: Config) -> list[str]:
    return ["flwr", "run", ".", "--stream", "--run-config",
            f"num_server_rounds={cfg.rounds} local_epochs={cfg.local_epochs} "
            f"min_clients={cfg.n_vehicles} fraction_fit=1.0"]


def _cmd_verify(_: Config) -> list[str]:
    return [PY, "-m", "pipeline.verify"]


STAGES: list[Stage] = [
    Stage("env", "Environment probe", False, _check_env, _cmd_env, est="~5 s"),
    Stage("dataset", "Download BDD100K", True, _check_dataset, _cmd_dataset, est="~10 min, 7.6 GB"),
    Stage("populate", "Populate shards", False, _check_populate, _cmd_populate, est="~1 min"),
    Stage("fleet", "Build vehicle fleet", False, _check_fleet, _cmd_fleet,
          cwd=paths.REPO, est="~30 s"),
    Stage("sanity", "Single-client GPU sanity", True, _check_sanity, _cmd_sanity, est="~2 min"),
    Stage("federate", "Federated run", True, _check_federate, _cmd_federate,
          data_root=paths.VEHICLE_ROOT, est="minutes to hours"),
    Stage("verify", "Verify pass criteria", False, _check_verify, _cmd_verify,
          cwd=paths.REPO, est="~5 s"),
]

BY_NAME = {s.name: s for s in STAGES}


def resolve(names: str | None) -> list[Stage]:
    """'fleet,federate' -> [Stage, Stage]; None -> the whole chain."""
    if not names:
        return list(STAGES)
    out = []
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        if n not in BY_NAME:
            raise SystemExit(f"unknown stage {n!r}; known: {', '.join(BY_NAME)}")
        out.append(BY_NAME[n])
    return out


def snapshot(cfg: Config) -> list[dict]:
    """Current state of every stage, for --list and for the control dashboard."""
    rows = []
    for s in STAGES:
        try:
            c = s.check(cfg)
        except Exception as e:                      # a broken check must not hide the stage
            c = Check(False, f"check failed: {e}")
        rows.append({"name": s.name, "title": s.title, "gated": s.gated,
                     "satisfied": c.satisfied, "detail": c.detail, "est": s.est})
    return rows
