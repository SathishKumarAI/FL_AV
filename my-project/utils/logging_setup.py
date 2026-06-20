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
        Path(os.path.dirname(log_file) or ".").mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
        )
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Attach our handler only once per logger; don't double-log via the root.
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False

    return logger
