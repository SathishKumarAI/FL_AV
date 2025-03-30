import os
import logging
import warnings
import numpy as np
import torch
import yaml
from pathlib import Path
from collections import OrderedDict
from ultralytics import YOLO

from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = configure_logging("task", "logs/task.log")

# Constants for model paths and URLs
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
DEFAULT_IMAGE_SIZE = 640  # Default image size for training/validation
DEFAULT_BATCH_ID_RANGE = (1, 10)  # Consistent with server and client

def download_model():
    """Download YOLO model if not found."""
    logger.info(f"[Task] Checking if model exists at {MODEL_PATH}...")

    # Ensure directory exists
    model_dir = os.path.dirname(MODEL_PATH) or "."  # Use current directory if empty
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(MODEL_PATH):  # Check before downloading
        logger.info(f"[Task] Model not found. Downloading from {MODEL_URL}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("[Task] Model downloaded successfully.")
        except Exception as e:
            logger.error(f"[Task] Failed to download the model from {MODEL_URL}: {e}", exc_info=True)
            raise RuntimeError(f"Cannot download a valid YOLO model to {MODEL_PATH}.") from e
    else:
        logger.info(f"[Task] Model already exists at {MODEL_PATH}, skipping download.")

def get_batch_path(batch_id):
    """
    Get the path to a batch directory.
    
    Args:
        batch_id: The batch ID
        
    Returns:
        Path: Path to the batch directory
    """
    return Path(f"batch/batch_{batch_id}")

def get_data_yaml_path(batch_id):
    """
    Get the path to a batch's data.yaml file.
    
    Args:
        batch_id: The batch ID
        
    Returns:
        Path: Path to the data.yaml file
    """
    return get_batch_path(batch_id) / "data.yaml"

# ----------------------------------------------------------
# 1) Configuration Loader
# ----------------------------------------------------------
def load_config(config_path):
    """
    Loads a YAML (or JSON) configuration file and returns a Python dict.

    :param config_path: Path to your config file (e.g. 'config.yaml').
    :return: Dictionary with parsed configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"[Task] Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"[Task] Failed to load config from {config_path}: {e}", exc_info=True)
        return {}


# ----------------------------------------------------------
# 2) Dataset Validation
# ----------------------------------------------------------
def validate_data_structure(batch_id, split="train"):
    """
    Checks if the directories for a given batch_id and data split exist
    and contain the necessary images (and labels, if not 'test').

    :param batch_id: Numeric or string ID for the dataset batch.
    :param split:    "train", "val", or "test".
    :return:         True if valid structure, else False.
    """
    base_path = get_batch_path(batch_id)
    split_path = base_path / split
    logger.info(f"[Task] Validating dataset structure for batch {batch_id}, split: {split}")

    if not split_path.exists():
        logger.error(f"[Task] Missing directory: {split_path}")
        return False

    img_path = split_path / "images"
    if not img_path.exists() or not any(img_path.iterdir()):
        logger.error(f"[Task] Missing or empty image directory: {img_path}")
        return False

    label_path = split_path / "labels" if split != "test" else None
    if label_path and (not label_path.exists() or not any(label_path.iterdir())):
        logger.error(f"[Task] Missing or empty label directory: {label_path}")
        return False
    
    yaml_path = get_data_yaml_path(batch_id)
    if not yaml_path.exists():
        logger.warning(f"[Task] Missing data.yaml file: {yaml_path}, you may need to create it.")
    
    logger.info(f"[Task] Validation successful for batch {batch_id}, split: {split}")
    return True


# ----------------------------------------------------------
# 3) GPU-based Batch Size Estimation
# ----------------------------------------------------------
def get_optimal_batch_size():
    """
    Attempts to pick a suitable batch size based on GPU free memory.
    You can override or extend logic as needed for large image datasets.
    """
    try:
        if torch.cuda.is_available():
            # mem_get_info()[0] -> total free memory on current GPU in bytes
            free_memory = torch.cuda.mem_get_info()[0]
            free_gb = free_memory / 1e9
            logger.info(f"[Task] Detected GPU with {free_gb:.2f} GB free memory.")

            # Simple heuristic for batch size
            if free_memory > 10e9:
                return 16
            elif free_memory > 6e9:
                return 8

        # Default fallback for CPU or smaller GPU memory
        return 4
    except Exception as e:
        logger.warning(f"[Task] Batch size detection failed: {str(e)}. Falling back to 4.")
        return 4


# ----------------------------------------------------------
# 5) Optional YOLO Training / Validation Helpers
# ----------------------------------------------------------
def train_custom(model: YOLO, device: str, epochs: int = 1, data_yaml: str = "data.yaml"):
    """
    An example custom training wrapper that calls YOLO's built-in train method.
    :param model:     YOLO model instance.
    :param device:    "cuda:0" or "cpu".
    :param epochs:    Number of local epochs to train.
    :param data_yaml: Path to the dataset config (YAML) describing train/val sets.
    :return:          Float representing the final training loss.
    """
    try:
        logger.info(f"[Task] Training YOLO model for {epochs} epoch(s) on device: {device}")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=DEFAULT_IMAGE_SIZE,
            device=device,
            verbose=False
        )
        final_loss = results.results_dict.get("train/loss", float("inf"))
        logger.info(f"[Task] Training completed with loss={final_loss}")
        return final_loss
    except Exception as e:
        logger.error(f"[Task] train_custom error: {e}", exc_info=True)
        return float("inf")


def test_custom(model: YOLO, device: str, data_yaml: str = "data.yaml"):
    """
    An example custom validation wrapper that calls YOLO's built-in val method.
    :param model:     YOLO model instance.
    :param device:    "cuda:0" or "cpu".
    :param data_yaml: Path to the dataset config (YAML) describing val set.
    :return:          (loss, mAP50) as float and float.
    """
    try:
        logger.info(f"[Task] Evaluating YOLO model on device: {device}")
        results = model.val(
            data=data_yaml,
            imgsz=DEFAULT_IMAGE_SIZE,
            device=device,
            verbose=False
        )
        loss = results.results_dict.get("val/loss", float("inf"))
        map50 = results.results_dict.get("metrics/mAP50", 0.0)
        logger.info(f"[Task] Validation completed: loss={loss}, mAP50={map50}")
        return float(loss), map50
    except Exception as e:
        logger.error(f"[Task] test_custom error: {e}", exc_info=True)
        return float("inf"), 0.0
