import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Rotation defaults (overridable via env for production tuning).
_MAX_BYTES = int(os.environ.get("FL_AV_LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
_BACKUP_COUNT = int(os.environ.get("FL_AV_LOG_BACKUP_COUNT", 5))
_LEVEL = os.environ.get("FL_AV_LOG_LEVEL", "INFO").upper()

#: `my-project/` — this file is `my-project/utils/logging_setup.py`.
#:
#: Every caller passes a *relative* path ("logs/server.log"), and every caller does it
#: at import time, so the log's location was decided by whichever directory the process
#: happened to start in. That is not a tidiness problem. `pytest my-project/tests` from
#: the repo root imports `my_project.server_app` during collection and thereby creates
#: an empty `logs/server.<pid>.log` in the **repo root**; that file then looked newer
#: than the real federation's log, and `pipeline.verify` reported "need >=2 rounds to
#: tell, saw 0" immediately after a six-round run had succeeded.
#:
#: Anchored here rather than at the five call sites: one resolution in the shared
#: function is a smaller change than five, and it covers the caller nobody has written
#: yet. An absolute `log_file` is still honoured unchanged — the tests pass tmp_path.


def project_root() -> Path:
    """Where a relative log path is resolved against.

    **`__file__` is not enough, and CI proved it.** `flwr run` does not execute the
    checkout: it copies the app and runs the copy —

        Successfully installed my-project to /home/runner/.flwr/apps/flower.my-project.1.0.0.480fd449

    — so `Path(__file__).parents[1]` is that copy. Anchoring on it alone sent the
    federation's logs into flwr's app cache, where the CI smoke's
    `glob("logs/server.*.log")` could not find them, and the run was judged to have
    aggregated nothing: `expected >=2 aggregate checksums, got []`. That is the same
    class of failure this whole file exists to stop, arrived at from the other side.

    So the order is:

    1. **`FL_AV_DATA_ROOT`** — the checkout that owns this run. `pipeline.paths.
       subprocess_env` sets it for every stage and the CI smoke sets it explicitly, so
       in practice this is the branch that is taken.
    2. **this file's own parent**, for a plain `pytest` or a direct import, where there
       is no run and no env.
    3. **the working directory**, but only when 2 lands inside `.flwr` — a copy is
       never the answer, and a log nobody can find is worse than one in the wrong
       place.
    """
    explicit = os.environ.get("FL_AV_DATA_ROOT")
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parents[1]
    return Path.cwd() if ".flwr" in here.parts else here


def configure_logging(logger_name, log_file=None):
    """
    Configure a module logger with a rotating file handler (or console).

    Production-friendly: log files rotate at FL_AV_LOG_MAX_BYTES (default 10 MB)
    keeping FL_AV_LOG_BACKUP_COUNT backups (default 5), so logs can't grow
    unbounded and fill the disk. Level is FL_AV_LOG_LEVEL (default INFO).

    Args:
        logger_name (str): The name of the logger.
        log_file (str, optional): Path to the log file. None -> console logging.
            A relative path is resolved by `project_root()`, never against the
            current working directory — see that function for what each cost.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, _LEVEL, logging.INFO))

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if log_file:
        # Ensure the log file's own directory exists (not a hardcoded "logs").
        path = Path(log_file)
        if not path.is_absolute():
            path = project_root() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        # One file per process. In simulation every client runs in its own Ray actor
        # process, and RotatingFileHandler is not multi-process safe: the writes
        # interleave and, worse, one process renaming the file mid-rotation silently
        # drops another's records. That made logs/client.log lie about which shard a
        # client trained on. Callers read logs/<name>.*.log (a glob), not one file.
        path = path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")
        handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
        )
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Attach our handler only once per logger; don't double-log via the root.
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False

    return logger
