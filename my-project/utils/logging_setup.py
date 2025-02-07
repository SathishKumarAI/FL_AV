import logging
from pathlib import Path

def configure_logging(logger_name, log_file=None):
    """
    Configures logging for the given module with an optional log file.
    
    Args:
        logger_name (str): The name of the logger.
        log_file (str, optional): Path to the log file. Defaults to None (console logging).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    # Create file handler if log_file is specified, otherwise use console logging
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger
