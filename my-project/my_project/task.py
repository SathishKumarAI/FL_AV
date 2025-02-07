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
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}", exc_info=True)
        return {}


# ----------------------------------------------------------
# 2) Dataset Validation
# ----------------------------------------------------------
def validate_data_structure(batch_id, split="train"):
    """
    Checks if the directories for a given batch_id and data split exist
    and contain the necessary images (and labels, if not 'test').

    Example path used below:
      C:/Users/sathish/Downloads/FL_ModelForAV/my-project/data/bdd100_batch/batch_{batch_id}/{split}/
        - images/
        - labels/ (if split != "test")

    :param batch_id: Numeric or string ID for the dataset batch.
    :param split:    "train", "val", or "test".
    :return:         True if valid structure, else False.
    """
    base_path = Path(f"C:/Users/sathish/Downloads/FL_ModelForAV/my-project/data/bdd100_batch/batch_{batch_id}")
    split_path = base_path / split
    logger.info(f"Validating dataset structure for batch {batch_id}, split: {split}")

    if not split_path.exists():
        logger.error(f"Missing directory: {split_path}")
        return False

    img_path = split_path / "images"
    if not img_path.exists() or not any(img_path.iterdir()):
        logger.error(f"Missing or empty image directory: {img_path}")
        return False

    label_path = split_path / "labels" if split != "test" else None
    if label_path and (not label_path.exists() or not any(label_path.iterdir())):
        logger.error(f"Missing or empty label directory: {label_path}")
        return False
    
    yaml_path = base_path / "data.yaml"
    if not yaml_path.exists():
        logger.warning(f"Missing data.yaml file: {yaml_path}, you may need to create it.")
    
    logger.info(f"Validation successful for batch {batch_id}, split: {split}")
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
            logger.info(f"Detected GPU with {free_gb:.2f} GB free memory.")

            # Simple heuristic for batch size
            if free_memory > 10e9:
                return 16
            elif free_memory > 6e9:
                return 8

        # Default fallback for CPU or smaller GPU memory
        return 4
    except Exception as e:
        logger.warning(f"Batch size detection failed: {str(e)}. Falling back to 4.")
        return 4


# ----------------------------------------------------------
# 4) YOLO Weight Extraction and Loading
# ----------------------------------------------------------
def get_weights(model: YOLO):
    """
    Extract YOLO model weights as a list of NumPy arrays.
    :param model: A loaded ultralytics.YOLO model instance.
    :return:      A list of NumPy arrays representing each parameter tensor.
    """
    try:
        logger.debug("Extracting YOLOv5 model weights...")
        weights_list = []
        for param in model.model.parameters():
            # Move to CPU before converting to numpy
            weights_list.append(param.detach().cpu().numpy())
        logger.debug(f"Extracted {len(weights_list)} weight tensors from model.")
        return weights_list
    except Exception as e:
        logger.error(f"get_weights error: {e}", exc_info=True)
        return []


def set_weights(model: YOLO, parameters):
    """
    Load a list of NumPy arrays (parameters) into the YOLO model's state_dict.
    :param model:       A loaded ultralytics.YOLO model instance.
    :param parameters:  A list of NumPy arrays that match model.model.state_dict() shape.
    :return:            Boolean indicating success or failure.
    """
    try:
        logger.debug("Setting YOLOv5 model weights...")
        state_dict = model.model.state_dict()
        new_state_dict = OrderedDict()

        idx = 0
        for name, param in state_dict.items():
            if idx < len(parameters):
                w = parameters[idx]
                # Check shape
                if param.shape == w.shape:
                    new_state_dict[name] = torch.from_numpy(w).to(param.device)
                else:
                    logger.warning(
                        f"Shape mismatch for {name}: expected {param.shape}, got {w.shape}. "
                        "Keeping original parameter."
                    )
                    new_state_dict[name] = param
                idx += 1
            else:
                # If fewer weights than state dict items
                logger.warning(f"No weight found for layer {name}, retaining original.")
                new_state_dict[name] = param

        # Load updated state dict
        model.model.load_state_dict(new_state_dict, strict=False)
        logger.debug("Model weights updated successfully.")
        return True
    except Exception as e:
        logger.error(f"set_weights error: {e}", exc_info=True)
        return False


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
        logger.info(f"Training YOLO model for {epochs} epoch(s) on device: {device}")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=1280,     # Example for large images
            device=device,
            verbose=False
        )
        final_loss = results.results_dict.get("train/loss", float("inf"))
        logger.info(f"Training completed with loss={final_loss}")
        return final_loss
    except Exception as e:
        logger.error(f"train_custom error: {e}", exc_info=True)
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
        logger.info(f"Evaluating YOLO model on device: {device}")
        results = model.val(
            data=data_yaml,
            imgsz=1280,     # Example for large images
            device=device,
            verbose=False
        )
        loss = results.results_dict.get("val/loss", float("inf"))
        map50 = results.results_dict.get("metrics/mAP50", 0.0)
        logger.info(f"Validation completed: loss={loss}, mAP50={map50}")
        return float(loss), map50
    except Exception as e:
        logger.error(f"test_custom error: {e}", exc_info=True)
        return float("inf"), 0.0
