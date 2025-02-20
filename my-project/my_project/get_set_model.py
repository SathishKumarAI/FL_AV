from typing import List
import warnings
import torch
import numpy as np
from ultralytics import  YOLO
# from ultralytics.models import Y
from collections import OrderedDict
from utils.logging_setup import configure_logging
import warnings
import logging
# def warn_with_log(message, category, filename, lineno, file=None, line=None):
#     logging.warning(f"{category.__name__}: {message} in {filename}:{lineno}")
# warnings.showwarning = warn_with_log

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("get_set", "logs/get_set.log")



# def get_weights(model: YOLO):
#     """
#     Extract YOLOv8 model weights as a list of NumPy arrays.
    
#     :param model: A loaded ultralytics.YOLO model instance.
#     :return:      A list of NumPy arrays representing each parameter tensor.
#     """
#     try:
#         logger.debug("Extracting YOLOv8 model weights...")

#         # Extract weights and convert to NumPy arrays
#         # weights_list = [param.data.cpu().numpy() for param in model["model"].state_dict().values()]
#         weights_list = [param.detach().cpu().numpy() for param in model["model"].parameters()]
#         logger.debug(f"Extracted {len(weights_list)} weight tensors from model.")
#         return weights_list
#     except Exception as e:
#         logger.error(f"get_weights error: {e}", exc_info=True)
#         return []



# def set_weights(model: YOLO, parameters: List[np.ndarray]) -> bool:

#     try:
#         logger.debug("Setting YOLOv8 model weights...")
# # def set_parameters(model: YOLO, parameters: List[np.ndarray]) -> None:
#         for i, param in enumerate(model["model"].parameters()):
#             param_ = torch.from_numpy(parameters[i]).to(param.device)
#             param.data.copy_(param_)
#         # # Access the underlying PyTorch model
#         # pytorch_model = model["model"]

#         # # Get the current state dictionary
#         # state_dict = pytorch_model.state_dict()

#         # # Ensure parameter count matches expected state_dict keys
#         # if len(parameters) != len(state_dict):
#         #     logger.warning(f"Expected {len(state_dict)} weight tensors, but got {len(parameters)}. "
#         #                 f"Check model compatibility.")
#         #     return False

#         # # Create a new state dictionary with updated weights
#         # new_state_dict = OrderedDict()

#         # for (name, param), new_weight in zip(state_dict.items(), parameters):
#         #     if not isinstance(new_weight, np.ndarray):
#         #         logger.warning(f"Parameter for {name} is not a NumPy array. Keeping original value.")
#         #         new_state_dict[name] = param
#         #         continue

#         #     # Convert NumPy array to PyTorch tensor
#         #     new_weight_tensor = torch.from_numpy(new_weight).to(param.device)

#         #     # Check shape compatibility
#         #     if param.shape != new_weight_tensor.shape:
#         #         logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {new_weight_tensor.shape}. "
#         #                 f"Skipping this layer.")
#         #         new_state_dict[name] = param  # Retain original weight
#         #     else:
#         #         new_state_dict[name] = new_weight_tensor  # Update weight

#         # # Load the updated state dictionary into the model
#         model.load_state_dict(param, strict=False)

#         logger.debug("Model weights updated successfully.")
#         return True

#     except Exception as e:
#         logger.error(f"Error in set_weights: {e}", exc_info=True)
#         return False

# Function to load YOLOv8 model with .pt weights
# def load_yolo_model(weight_path=r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov8s.pt"):
#     """
#     Load a YOLOv8 model from a .pt file.
#     :param weight_path: Path to the .pt weight file.
#     :return:            Loaded YOLOv8 model.
#     """
#     try:
#         logger.debug(f"Loading YOLOv8 model from {weight_path}...")
#         # model = YOLO('yolov8n.yaml').load('yolov8n.pt')
#         # model = YOLO(r'C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov8.yaml').load(r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov8s.pt")
#         # model['model'].nc = 13  #
#         # model = torch.load(r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolo8n.yaml")

#         # model = torch.load(r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolo8n.yaml")
#         # model = YOLO(r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolo8n.yaml").load()
#         # model = load_yolo_model("yolov8.pt")

#         # Create the YOLOv8 model from your custom .yaml file
#         model = YOLO(r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolo8n.yaml")

#         # Load the pretrained weights
#         model.load_weights(r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov8s.pt")

#         # Start training with your data
#         # model.train(data="data.yaml")
#         if model is None:
#             raise RuntimeError("Failed to load YOLOv8 model.")

#         logger.debug("Model loaded successfully.")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load YOLOv8 model: {e}", exc_info=True)
#         return None

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
    """
    Load a YOLOv8 model from a .yaml file, set number of classes, and return the model.
    
    :param yaml_path: Path to the .yaml configuration file.
    :param weight_path: Path to the .pt weight file.
    :return: The actual PyTorch model object.
    """
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