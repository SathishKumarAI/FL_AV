"""Push pipeline facts into MLflow. MLflow owns storage, history and charts.

Two writers land in the same experiment and that is intentional:

* Ultralytics' own MLflow callback logs *per-vehicle training* (box/cls/dfl loss,
  mAP curves) with no code from us -- it ships in ultralytics 8.4.
* This module logs *federation-level* facts that no single training run can see:
  the round-over-round aggregate checksum, which vehicle held which shard, and the
  GPU energy the whole thing cost.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from . import logparse, paths

EXPERIMENT = paths.MLFLOW_EXPERIMENT


def ensure_experiment() -> None:
    """Create the experiment, with its artifact location named, before anything logs.

    Order matters and it is the whole reason this is callable on its own. Ultralytics'
    callback calls ``set_experiment`` too, and whoever gets there first decides where
    artifacts live. If that is the callback -- running with ``cwd=my-project`` and no
    artifact location -- MLflow resolves them against the CWD and a federation leaves
    24 run directories under ``my-project/mlruns/``, outside the store the sink writes
    to. The pipeline therefore creates the experiment before it launches a stage.
    """
    _mlflow()


def _mlflow():
    import mlflow  # lazy: importing mlflow costs ~2s, and --list should stay instant
    mlflow.set_tracking_uri(paths.mlflow_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        # Named explicitly. A database backend does not imply where artifacts go, and
        # the default resolves against the CWD -- which for this project means one
        # store per place a subprocess was launched from.
        paths.MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=paths.MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


@contextlib.contextmanager
def run(name: str, params: dict | None = None, nested: bool = False):
    """A run (or nested run) that always closes, even when a stage raises."""
    mlflow = _mlflow()
    with mlflow.start_run(run_name=name, nested=nested) as active:
        if params:
            # MLflow rejects non-scalar params; stringify rather than lose them.
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield active


def log_fleet(fleet: list[dict]) -> None:
    """Vehicle -> condition -> shard size. The 'inputs' half of the final report."""
    mlflow = _mlflow()
    for v in fleet:
        vid = v["vid"]
        mlflow.log_param(f"vehicle_{vid}_condition", v["condition"])
        mlflow.log_metric(f"vehicle_{vid}_n_train", v.get("n_train", len(v.get("train", []))))
        mlflow.log_metric(f"vehicle_{vid}_n_val", v.get("n_val", len(v.get("val", []))))


def log_federation(log_dir: Path, metrics_csv: Path) -> dict:
    """Per-round aggregate checksums + aggregated metrics. Returns what it logged."""
    mlflow = _mlflow()
    checksums = logparse.aggregate_checksums(log_dir)
    for i, c in enumerate(checksums, start=1):
        mlflow.log_metric("aggregate_checksum", c, step=i)

    rows = logparse.read_metrics_csv(metrics_csv)
    for row in rows:
        step = int(row.get("round") or 0)
        stage = row.get("stage", "")
        for key in ("loss", "precision", "recall", "mAP50", "mAP50-95", "fitness"):
            val = row.get(key)
            if val is not None:
                mlflow.log_metric(f"{stage}_{key}".replace("-", "_"), float(val), step=step)

    learned, detail = logparse.federation_learned(log_dir)
    mlflow.set_tag("federation_learned", str(learned))
    mlflow.set_tag("federation_detail", detail)
    return {"checksums": checksums, "rows": rows, "learned": learned, "detail": detail}


def log_gpu(summary: dict, prefix: str = "gpu") -> None:
    """Energy, peak VRAM, peak power. The number nobody logs and everybody asks for."""
    mlflow = _mlflow()
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{prefix}_{k}", float(v))


def log_artifact(path: Path, subdir: str | None = None) -> None:
    if path.exists():
        _mlflow().log_artifact(str(path), artifact_path=subdir)
