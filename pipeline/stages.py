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
    ray_address: str | None = None   # set => attach to an existing head node

    @property
    def per_vehicle(self) -> int:
        return 300 if self.profile == "demo" else 6308

    @property
    def imgsz(self) -> int:
        """Image size for the *sanity* stage only.

        The federation's size is my-project's DEFAULT_IMAGE_SIZE (640) and is not
        reachable from here -- changing it would mean editing client_app.py, which
        this component is not allowed to do. So `demo` speeds things up through
        fewer images per vehicle, not smaller ones.
        """
        return 320 if self.profile == "demo" else 640

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(per_vehicle=self.per_vehicle, imgsz=self.imgsz)
        return d


@dataclass
class Check:
    satisfied: bool
    detail: str


# flwr can exit 0 having crashed: it prints "Simulation Runtime crashed" and
# "Exit Code: 700" and still returns success. Trusting the exit code alone marked a
# dead run "ok" -- exactly the silent-failure pattern this project keeps producing --
# so the output is scanned too.
CRASH_MARKERS = (
    "Simulation Runtime crashed",
    "An error was encountered. Ending simulation",
    "Simulation raised an exception",
    "Traceback (most recent call last)",
)


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
    crash_markers: tuple[str, ...] = ()   # output that means failure despite exit 0


def scan_for_crash(lines: list[str], markers: tuple[str, ...]) -> str | None:
    """Return the first crash marker present in the output, or None."""
    for marker in markers:
        for line in lines:
            if marker in line:
                return marker
    return None


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
    # A shard must exist for every id the server can pick, not just for the clients
    # that will run -- DEFAULT_BATCH_ID_RANGE is (1, 10) and is not configurable.
    want = max(cfg.n_vehicles, len(list(paths.BATCH_IDS)))
    fleet = vehicles.load_fleet()
    if len(fleet) < want:
        return Check(False, f"fleet has {len(fleet)} shards, need {want} "
                            f"(the server can assign any id in {tuple(paths.BATCH_IDS)[0]}.."
                            f"{tuple(paths.BATCH_IDS)[-1]})")
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
    # num-supernodes MUST track the vehicle count. It lives in the federation config,
    # which flwr migrates out of pyproject.toml into ~/.flwr/config.toml on first run
    # -- so editing pyproject would be silently ignored and the run would hang
    # forever waiting for clients that never arrive. Override it on the CLI instead.
    fed = f"num-supernodes={cfg.n_vehicles} client-resources-num-gpus=1.0 client-resources-num-cpus=2"
    if not cfg.ray_address:
        # Ray refuses num_cpus/num_gpus when attaching to a cluster that already
        # exists ("When connecting to an existing cluster, num_cpus and num_gpus must
        # not be provided"), so these are only valid when flwr starts Ray itself.
        fed += " init-args-num-gpus=1 init-args-num-cpus=8"
    return ["flwr", "run", ".", "--stream", "--federation-config", fed,
            "--run-config",
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
          data_root=paths.VEHICLE_ROOT, est="minutes to hours",
          crash_markers=CRASH_MARKERS),
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
