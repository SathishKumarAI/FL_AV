from typing import List, Optional
import warnings
import torch
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("get_set", "logs/get_set.log")

# Constants
DEFAULT_NUM_CLASSES = 13  # The default number of classes for our model

def get_weights(model):
    """
    Extract YOLOv8 model weights as a list of NumPy arrays.
    
    Args:
        model: A loaded YOLOv8 model instance (DetectionModel).
    
    Returns:
        A list of NumPy arrays representing each parameter tensor, or empty list on error.
    """
    try:
        logger.debug("[GetSet] Extracting YOLOv8 model weights...")
        # The model is now a DetectionModel object, not a dictionary
        weights_list = [param.detach().cpu().numpy() for param in model.parameters()]
        
        # Calculate checksum for debugging
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)
        
        logger.debug(f"[GetSet] Extracted {len(weights_list)} weight tensors with checksum: {weights_checksum}")
        return weights_list
    except Exception as e:
        logger.error(f"[GetSet] get_weights error: {e}", exc_info=True)
        return []


def set_weights(model, parameters: List[np.ndarray]) -> bool:
    """
    Apply weights to YOLOv8 model parameters.
    
    Args:
        model: The YOLOv8 model instance
        parameters: List of NumPy arrays with weights
        
    Returns:
        bool: True on success, False on error
    """
    try:
        logger.debug("[GetSet] Setting YOLOv8 model weights...")
        
        # Calculate checksum for debugging
        weights_checksum = sum(w.sum() for w in parameters if w.size > 0)
        logger.debug(f"[GetSet] Applying weights with checksum: {weights_checksum}")
        
        # Validate parameter length
        model_params = list(model.parameters())
        if len(parameters) != len(model_params):
            logger.error(f"[GetSet] Parameter count mismatch: model has {len(model_params)} layers, but received {len(parameters)} arrays")
            return False
        
        # Apply weights to each layer
        for i, param in enumerate(model_params):
            param_ = torch.from_numpy(parameters[i]).to(param.device)
            
            # Check shape compatibility
            if param_.shape != param.shape:
                logger.error(
                    f"[GetSet] Shape mismatch at parameter {i}: "
                    f"model shape {param.shape} != parameter shape {param_.shape}"
                )
                return False
                
            # Copy the weights
            param.data.copy_(param_)
            
        logger.debug("[GetSet] Model weights updated successfully.")
        return True
    except Exception as e:
        logger.error(f"[GetSet] Error in set_weights: {e}", exc_info=True)
        return False


def load_yolo_model(yaml_path="models/yolo8n.yaml", 
                weight_path="models/yolov8s.pt") -> Optional[torch.nn.Module]:
    """
    Load a YOLOv8 model with specified configuration and weights.
    
    Args:
        yaml_path: Path to the model configuration YAML
        weight_path: Path to the model weights
        
    Returns:
        The loaded model or None on error
    """
    try:
        logger.debug(f"[GetSet] Loading YOLOv8 model from {yaml_path} with weights from {weight_path}...")
        
        # Check if files exist
        import os
        if not os.path.exists(yaml_path):
            logger.error(f"[GetSet] Model config not found: {yaml_path}")
            return None
            
        if weight_path and not os.path.exists(weight_path):
            logger.warning(f"[GetSet] Model weights not found: {weight_path}, will use default initialization")
        
        # Create the base YOLO model from yaml
        yolo = YOLO(yaml_path)
        
        # Load weights if provided
        if weight_path and os.path.exists(weight_path):
            yolo.load(weight_path)
        
        # Access the actual PyTorch model
        model = yolo.model
        
        # Set number of classes
        model.nc = DEFAULT_NUM_CLASSES
        
        # Adjust the detection layers if needed for the new class count
        if hasattr(model, 'head'):
            model.head.nc = DEFAULT_NUM_CLASSES
        
        logger.debug(f"[GetSet] Model loaded successfully with nc={DEFAULT_NUM_CLASSES}.")
        return model
    except Exception as e:
        logger.error(f"[GetSet] Failed to load YOLOv8 model: {e}", exc_info=True)
        return None