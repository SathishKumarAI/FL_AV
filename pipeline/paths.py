"""Where things live, and the environment every stage subprocess needs.

Single place that knows the repo layout, so the rest of the package never
hardcodes a path or an env var that has already bitten someone.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "my-project"          # invoked, never modified
HERE = Path(__file__).resolve().parent
STATE = HERE / ".state"            # markers + attribute cache; gitignored
MLFLOW_STORE = HERE / "mlruns"     # gitignored
REPORTS = HERE / "reports"         # gitignored
# Vehicle shards live here, NOT in my-project. task.get_batch_path() resolves
# FL_AV_DATA_ROOT/batch/batch_<id>, so pointing that env var at VEHICLE_ROOT is all
# it takes to run the federation on simulated vehicles without touching my-project.
VEHICLE_ROOT = HERE / "vehicles"
VEHICLE_BATCHES = VEHICLE_ROOT / "batch"

BATCH_IDS = range(1, 11)

# Where kagglehub puts solesensei/solesensei_bdd100k. The archive nests a second
# bdd100k inside the first -- that is the archive's own layout, not a typo.
KAGGLE_DATASET = "solesensei/solesensei_bdd100k"
_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "solesensei" / "solesensei_bdd100k"


def find_pool() -> Path | None:
    """Locate the images/100k directory holding train/ and val/, or None."""
    if not _CACHE.is_dir():
        return None
    for candidate in sorted(_CACHE.glob("versions/*/**/images/100k")):
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    return None


LABELS_DIR_NAME = "bdd100k_labels_release"


def find_label_jsons() -> list[Path]:
    """The raw BDD100K attribute JSONs (weather/scene/timeofday), if downloaded."""
    if not _CACHE.is_dir():
        return []
    return sorted(_CACHE.glob(f"versions/*/{LABELS_DIR_NAME}/**/bdd100k_labels_images_*.json"))


def log_dirs() -> list[Path]:
    """Every directory a run's logs can land in.

    my-project's loggers use CWD-relative paths and are configured at import time.
    Depending on whether a component runs as a direct subprocess (cwd=my-project) or
    inside a Ray worker (which inherits the head node's cwd), one run can scatter its
    logs across both. Looking in only one place made a working federation report as a
    failure.
    """
    return [d for d in (PROJECT / "logs", REPO / "logs") if d.is_dir()]


def subprocess_env(ray_address: str | None = None, data_root: Path | None = None) -> dict:
    """Environment for every stage subprocess.

    Two of these are load-bearing and cost hours when missing:

    * ``FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION`` -- without it flwr >= 1.31
      builds its own runtime env via ``uv sync`` and installs the CPU-only torch
      wheel, so every client trains on CPU at ~5.5x the wall clock, silently.
    * ``FL_AV_DATA_ROOT`` -- pins the shard root to this checkout rather than the
      path baked into the committed ``data.yaml``.
    """
    env = os.environ.copy()
    env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    env["FL_AV_DATA_ROOT"] = str(data_root or PROJECT)
    env.setdefault("MLFLOW_TRACKING_URI", MLFLOW_STORE.as_uri())
    if ray_address:
        # Makes flwr's ray.init() attach to an already-running head node instead
        # of creating its own with include_dashboard=False hardcoded.
        env["RAY_ADDRESS"] = ray_address
    else:
        # Explicitly clear it. Inheriting a stale RAY_ADDRESS makes flwr attach to a
        # cluster it did not configure, and Ray then rejects the num_cpus/num_gpus
        # that flwr passes: "When connecting to an existing cluster, num_cpus and
        # num_gpus must not be provided."
        env.pop("RAY_ADDRESS", None)
    return env


def count_files(directory: Path, suffixes: set[str] | None = None) -> int:
    """Count files in a directory without materialising the list."""
    if not directory.is_dir():
        return 0
    n = 0
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and (suffixes is None or Path(entry.name).suffix.lower() in suffixes):
                n += 1
    return n
