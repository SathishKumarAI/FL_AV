import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Rotation defaults (overridable via env for production tuning).
_MAX_BYTES = int(os.environ.get("FL_AV_LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
_BACKUP_COUNT = int(os.environ.get("FL_AV_LOG_BACKUP_COUNT", 5))
_LEVEL = os.environ.get("FL_AV_LOG_LEVEL", "INFO").upper()


def configure_logging(logger_name, log_file=None):
    """
    Configure a module logger with a rotating file handler (or console).

    Production-friendly: log files rotate at FL_AV_LOG_MAX_BYTES (default 10 MB)
    keeping FL_AV_LOG_BACKUP_COUNT backups (default 5), so logs can't grow
    unbounded and fill the disk. Level is FL_AV_LOG_LEVEL (default INFO).

    Args:
        logger_name (str): The name of the logger.
        log_file (str, optional): Path to the log file. None -> console logging.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, _LEVEL, logging.INFO))

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if log_file:
        # Ensure the log file's own directory exists (not a hardcoded "logs").
        path = Path(log_file)
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
