import os
import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from collections import OrderedDict
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from utils.logging_setup import configure_logging  # Ensure logging setup

logger = configure_logging("task", "logs/task.log")

# Function to validate dataset structure




# Function to validate dataset structure
def validate_data_structure(batch_id, split="train"):
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
        logger.warning(f"Missing data.yaml file: {yaml_path}, attempting to create one.")
        # create_data_yaml(yaml_path, base_path)
    
    logger.info(f"Validation successful for batch {batch_id}, split: {split}")
    return True


# Function to determine optimal batch size
def get_optimal_batch_size():
    try:
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0]  # Free GPU memory
            logger.info(f"Detected GPU with {free_memory / 1e9:.2f} GB free memory.")
            if free_memory > 10e9:
                return 16
            elif free_memory > 6e9:
                return 8
        return 4  # Default for CPU
    except Exception as e:
        logger.warning(f"Batch size detection failed: {str(e)}")
        return 4

# Function to load data
def load_data(batch_id, split="train"):
    logger.info(f"Loading data for batch {batch_id}, split: {split}.")

    # if not validate_data_structure(batch_id, split):
    #     logger.error(f"Validation failed for batch {batch_id}. Cannot load data.")
    #     return None

    base_path = Path(f"C:/Users/sathish/Downloads/FL_ModelForAV/my-project/data/bdd100_batch/batch_{batch_id}")
    yaml_path = base_path / "data.yaml"
    
    try:
        dataset = YOLODataset(yaml_path.as_posix(), task="detect")

        dataloader = DataLoader(
            dataset,
            batch_size=get_optimal_batch_size(),
            shuffle=(split == "train"),
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )
        logger.error(f"Failed to load dataset for batch .se53 Error: {str(e)}")
        logger.info(f"DataLoader successfully created for batch {batch_id}, split: {split}.")
        return dataloader

    except Exception as e:
        logger.error(f"Failed to load dataset for batch {batch_id}, split: {split}. Error: {str(e)}")
        return None


# Function to extract model weights
def get_weights(model):
    try:
        logger.info("Extracting model weights...")
        weights = [param.detach().cpu().numpy() for param in model.model.parameters()]
        logger.info("Model weights successfully extracted.")
        return weights
    except Exception as e:
        logger.error(f"Weight extraction failed: {e}")
        return None

# Function to set model weights
def set_weights(model, weights):
    try:
        logger.info("Setting model weights...")
        state_dict = OrderedDict()
        current_state = model.model.state_dict()
        mismatch_count = 0

        trainable_params = [p for p in current_state.values() if p.requires_grad]
        if len(weights) != len(trainable_params):
            logger.warning(f"Mismatch in weight count: Expected {len(trainable_params)}, got {len(weights)}")

        for (name, param), new_weight in zip(current_state.items(), weights):
            if not param.requires_grad:
                continue

            if param.shape == new_weight.shape:
                state_dict[name] = torch.from_numpy(new_weight)
            else:
                mismatch_count += 1
                logger.warning(f"Shape mismatch for {name}: Expected {param.shape}, got {new_weight.shape}")

        if mismatch_count > 0:
            logger.warning(f"Total mismatched layers: {mismatch_count}")

        model.model.load_state_dict(state_dict, strict=False)
        logger.info("Model weights successfully updated.")
        return True
    except Exception as e:
        logger.error(f"Weight setting failed: {e}")
        return False

# Function to train the model
def train(model, device):
    try:
        logger.info(f"Starting training for  epochs on device: {device}")
        # for epoch in range(epochs):
        # logger.info(f"Epoch {epoch + 1}/{epochs}...")
        results = model.train(
            data='data.yaml',
            epochs=1,# Train one epoch at a time
            imgsz=640,                
            device=device,
            verbose=False
        )
        # loss = results.results_dict.get("train/loss", float("inf"))
        # logger.info(f"Epoch {epoch + 1} completed. Loss: {loss:.4f}")
        # logger.info(f"Epoch {epoch + 1} completed.")
        logger.info("Training completed successfully.")
        return results
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return float("inf")

# Function to test the model
def test(model, device):
    try:
        logger.info(f"Starting validation on device: {device}")
        metrics = model.predict(
            data='data_yaml',
            device=device,
            verbose=False,
            conf = 0.5
            
        )
        # loss = metrics.results_dict.get('val/loss', float("inf"))
        # mAP = metrics.results_dict.get('metrics/mAP50-95', 0.0)

        # logger.info(f"Validation completed. Loss: {loss:.4f}, mAP50-95: {mAP:.4f}")
        
        
        return 
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return float("inf"), 0.0


import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config