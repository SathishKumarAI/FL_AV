import logging
import functools
from typing import Callable
from ultralytics import YOLO

def configure_logging(filename: str = "flwr.log"):
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_yolov5_model(model_path: str = "yolov5s.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        raise

def log_exception(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper