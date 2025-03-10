from typing import List
import warnings
import torch
import numpy as np
from ultralytics import  YOLO
from collections import OrderedDict
from utils.logging_setup import configure_logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("get_set", "logs/get_set.log")


def get_weights(model):
    """
    Extract YOLOv8 model weights as a list of NumPy arrays.
    
    :param model: A loaded YOLOv8 model instance (DetectionModel).
    :return: A list of NumPy arrays representing each parameter tensor.
    """
    try:
        logger.debug("Extracting YOLOv8 model weights...")
        # The model is now a DetectionModel object, not a dictionary
        weights_list = [param.detach().cpu().numpy() for param in model.parameters()]
        logger.debug(f"Extracted {len(weights_list)} weight tensors from model.")
        return weights_list
    except Exception as e:
        logger.error(f"get_weights error: {e}", exc_info=True)
        return []


def set_weights(model, parameters: List[np.ndarray]) -> bool:
    try:
        logger.debug("Setting YOLOv8 model weights...")
        # The model is now a DetectionModel object, not a dictionary
        for i, param in enumerate(model.parameters()):
            param_ = torch.from_numpy(parameters[i]).to(param.device)
            param.data.copy_(param_)
            
        logger.debug("Model weights updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error in set_weights: {e}", exc_info=True)
        return False


def load_yolo_model(yaml_path=r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolo8n.yaml", 
                weight_path=r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov8s.pt"):

    try:
        logger.debug(f"Loading YOLOv8 model from {yaml_path} with weights from {weight_path}...")
        
        # Create the base YOLO model from yaml
        yolo = YOLO(yaml_path)
        
        # Load weights if provided
        if weight_path:
            yolo.load(weight_path)
        
        # Access the actual PyTorch model
        model = yolo.model
        
        # Set number of classes to 13
        model.nc = 13
        
        # Adjust the detection layers if needed for the new class count
        if hasattr(model, 'head'):
            model.head.nc = 13
        
        logger.debug("Model loaded successfully with nc=13.")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}", exc_info=True)
        return None