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

from . import baseline, holdout, paths, validate, vehicles

PY = sys.executable

#: Aggregation strategies this pipeline offers, mirroring server_app.STRATEGIES.
#: Mirrored rather than imported: reaching into my-project would drag flwr and
#: ultralytics into a package that is meant to stay stdlib-light. The two lists are
#: asserted equal by my-project/tests/test_strategy_registry.py, so they cannot drift.
STRATEGIES = ("fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad", "fedavgm",
              "fedmedian", "fedtrimmedavg", "krum", "bulyan", "qfedavg",
              "faulttolerantfedavg")


@dataclass
class Config:
    """Everything a run is parameterised by. Also the 'inputs' half of the report."""

    profile: str = "demo"        # demo | full
    n_vehicles: int = 6
    rounds: int = 2
    local_epochs: int = 1
    seed: int = 0
    partition: str = "condition"     # any key of vehicles.PARTITIONERS
    strategy: str = "fedavg"         # any key of server_app.STRATEGIES
    proximal_mu: float = 0.0         # FedProx only; >0 turns the proximal term on
    holdout_size: int = 1000         # images no vehicle may train or self-evaluate on
    alpha: float = 0.5               # dirichlet only: smaller = more skewed
    per_vehicle_override: int = 0    # 0 = use the profile default
    ray_address: str | None = None   # set => attach to an existing head node

    @property
    def per_vehicle(self) -> int:
        """Images per vehicle.

        Worth overriding for condition partitioning: BDD100K only holds ~5 800 rainy
        and ~6 300 snowy images in total, so asking for 6 308 per vehicle exhausts the
        condition and the shard tops up with whatever is left -- which quietly turns a
        non-IID run into a nearly-IID one. Keep it below the rarest condition's count
        if the bias is meant to be real.
        """
        if self.per_vehicle_override:
            return self.per_vehicle_override
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
    meta = vehicles.load_fleet_meta()
    held = len(holdout.names())
    if meta:
        # The manifest says how this fleet was built, so intent is compared rather
        # than guessed. Alpha counts only where it means something.
        differs = [k for k, want in (("partition", cfg.partition), ("seed", cfg.seed),
                                     ("per_vehicle", cfg.per_vehicle)) if meta.get(k) != want]
        if cfg.partition == "dirichlet" and meta.get("alpha") != cfg.alpha:
            differs.append("alpha")
        # A fleet built against a different holdout may hold images the global model
        # is about to be scored on. That would make the one honest metric partly
        # self-referential, so it forces a rebuild.
        if meta.get("holdout", 0) != held:
            differs.append("holdout")
        if differs:
            return Check(False, "fleet on disk differs in " + ", ".join(differs) +
                                f" (built {meta.get('partition')!r}, want {cfg.partition!r})")
    else:
        # Pre-manifest fleet: fall back to the old inference, which can only
        # distinguish random from everything else.
        if held:
            return Check(False, f"fleet predates the {held}-image holdout and cannot be shown "
                                f"to exclude it; rebuild")
        want_random = cfg.partition == "random"
        is_random = all(v.get("condition") == "random mix" for v in fleet)
        if want_random != is_random:
            return Check(False, f"fleet was built with a different partition than {cfg.partition!r}")

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
            "--seed", str(cfg.seed),
            "--partition", cfg.partition,
            "--alpha", str(cfg.alpha)]


#: One epoch on one shard, through the Python API rather than a console script.
#: `python -m ultralytics.cfg` worked until ultralytics 8.4 made cfg a package with
#: no __main__, and the `yolo` console script is not guaranteed to be on PATH for the
#: interpreter running the pipeline. Calling the same API the clients call cannot
#: drift from what a client does, which is the point of a sanity stage.
_SANITY = """
from ultralytics import YOLO
YOLO({model!r}).load({pretrained!r}).train(
    data={data!r}, epochs=1, imgsz={imgsz}, batch=4, device={device!r}, workers=2,
    plots=False, project="runs/pipeline", name="sanity", exist_ok=True)
"""


def _cmd_sanity(cfg: Config) -> list[str]:
    return [PY, "-c", _SANITY.format(
        model="models/yolov8s-13.yaml", pretrained="models/yolov8s.pt",
        data="batch/batch_1/data.runtime.yaml", imgsz=cfg.imgsz, device="0")]


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
            f'num_server_rounds={cfg.rounds} local_epochs={cfg.local_epochs} '
            f'min_clients={cfg.n_vehicles} fraction_fit=1.0 '
            # Quoted because flwr parses run-config values as TOML: an unquoted
            # fedadam is a bare word, not a string, and the run dies on parse.
            f'strategy="{cfg.strategy}" proximal_mu={cfg.proximal_mu}']


def _cmd_verify(_: Config) -> list[str]:
    return [PY, "-m", "pipeline.verify"]


def _check_holdout(cfg: Config) -> Check:
    info = holdout.meta()
    if not info:
        return Check(False, "not yet carved")
    if info.get("size") != cfg.holdout_size or info.get("seed") != cfg.seed:
        return Check(False, f"holdout on disk is size={info.get('size')} seed={info.get('seed')}, "
                            f"config wants size={cfg.holdout_size} seed={cfg.seed}")
    return Check(True, f"{info.get('linked')} images held out, no vehicle sees them")


def _cmd_holdout(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.holdout", "--build",
            "--size", str(cfg.holdout_size), "--seed", str(cfg.seed)]


def _check_validate(_: Config) -> Check:
    # Never satisfied: it is seconds of scanning, and the whole point is to run it
    # against the shards a federation is about to train on, not against a memory of
    # the last time they were sound.
    return Check(False, "always runs; nothing measured on bad shards is worth reading")


def _cmd_validate(_: Config) -> list[str]:
    return [PY, "-m", "pipeline.validate"]


def _check_evaluate(_: Config) -> Check:
    return Check(False, "always runs; the only honest global metric")


def _cmd_evaluate(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.holdout", "--evaluate", "--imgsz", str(cfg.imgsz)]


def _check_baseline(cfg: Config) -> Check:
    row = baseline.result()
    if not row:
        return Check(False, "no centralised run yet; federated numbers have no scale without it")
    want = cfg.rounds * cfg.local_epochs
    if row.get("epochs") != want:
        return Check(False, f"baseline was {row.get('epochs')} epochs, this run's budget is {want}")
    return Check(True, f"mAP50 {row['mAP50']:.4f} on {row['images']} pooled images")


def _cmd_baseline(cfg: Config) -> list[str]:
    return [PY, "-m", "pipeline.baseline", "--rounds", str(cfg.rounds),
            "--local-epochs", str(cfg.local_epochs), "--imgsz", str(cfg.imgsz)]


STAGES: list[Stage] = [
    Stage("env", "Environment probe", False, _check_env, _cmd_env, est="~5 s"),
    Stage("dataset", "Download BDD100K", True, _check_dataset, _cmd_dataset, est="~10 min, 7.6 GB"),
    Stage("populate", "Populate shards", False, _check_populate, _cmd_populate, est="~1 min"),
    # Before fleet, always. A holdout carved afterwards is already inside somebody's
    # val split, and the "global" metric measured on it would be partly
    # self-referential -- a silent failure of exactly the kind this project collects.
    Stage("holdout", "Carve the shared holdout", False, _check_holdout, _cmd_holdout,
          cwd=paths.REPO, est="~20 s"),
    Stage("fleet", "Build vehicle fleet", False, _check_fleet, _cmd_fleet,
          cwd=paths.REPO, est="~30 s"),
    Stage("validate", "Validate the shards", False, _check_validate, _cmd_validate,
          cwd=paths.REPO, est="~10 s"),
    Stage("sanity", "Single-client GPU sanity", True, _check_sanity, _cmd_sanity, est="~2 min"),
    Stage("federate", "Federated run", True, _check_federate, _cmd_federate,
          data_root=paths.VEHICLE_ROOT, est="minutes to hours",
          crash_markers=CRASH_MARKERS),
    Stage("evaluate", "Score the global model on the holdout", False,
          _check_evaluate, _cmd_evaluate, cwd=paths.REPO, est="~10 s per round"),
    Stage("verify", "Verify pass criteria", False, _check_verify, _cmd_verify,
          cwd=paths.REPO, est="~5 s"),
    # Last, and gated: it trains a whole model. Not part of --all by accident.
    Stage("baseline", "Centralised ceiling on pooled data", True,
          _check_baseline, _cmd_baseline, cwd=paths.REPO, est="as long as one full run"),
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
