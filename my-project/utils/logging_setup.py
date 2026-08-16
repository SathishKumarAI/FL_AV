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
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_logging(logger_name, log_file=None):
    """
    Configure a module logger with a rotating file handler (or console).

    Production-friendly: log files rotate at FL_AV_LOG_MAX_BYTES (default 10 MB)
    keeping FL_AV_LOG_BACKUP_COUNT backups (default 5), so logs can't grow
    unbounded and fill the disk. Level is FL_AV_LOG_LEVEL (default INFO).

    Args:
        logger_name (str): The name of the logger.
        log_file (str, optional): Path to the log file. None -> console logging.
            A relative path is resolved against `my-project/`, never against the
            current working directory — see _PROJECT_ROOT above for what that cost.

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
            path = _PROJECT_ROOT / path
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
