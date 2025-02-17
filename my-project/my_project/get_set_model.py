import warnings
import torch
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from utils.logging_setup import configure_logging
import warnings
import logging
# def warn_with_log(message, category, filename, lineno, file=None, line=None):
#     logging.warning(f"{category.__name__}: {message} in {filename}:{lineno}")
# warnings.showwarning = warn_with_log

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("get_set", "logs/get_set.log")



def get_weights(model: YOLO):
    """
    Extract YOLOv8 model weights as a list of NumPy arrays.
    
    :param model: A loaded ultralytics.YOLO model instance.
    :return:      A list of NumPy arrays representing each parameter tensor.
    """
    try:
        logger.debug("Extracting YOLOv8 model weights...")

        # Extract weights and convert to NumPy arrays
        weights_list = [param.data.cpu().numpy() for param in model.model.state_dict().values()]

        logger.debug(f"Extracted {len(weights_list)} weight tensors from model.")
        return weights_list
    except Exception as e:
        logger.error(f"get_weights error: {e}", exc_info=True)
        return []



def set_weights(model: YOLO, parameters):
    """
    Load a list of NumPy arrays (parameters) into the YOLOv8 model's state_dict.
    
    :param model:       A loaded ultralytics.YOLO model instance.
    :param parameters:  A list of NumPy arrays that match model.model.state_dict() shape.
    :return:            Boolean indicating success or failure.
    """
    try:
        logger.debug("Setting YOLOv8 model weights...")

        state_dict = model.model.state_dict()
        new_state_dict = OrderedDict()

        # Ensure parameter count matches expected state_dict keys
        if len(parameters) != len(state_dict):
            logger.warning(f"Expected {len(state_dict)} weight tensors, but got {len(parameters)}. "
                        f"Check model compatibility.")
            return False

        # Assign new weights
        for (name, param), w in zip(state_dict.items(), parameters):
            if isinstance(w, np.ndarray):  # Ensure the weight is a NumPy array
                w_tensor = torch.from_numpy(w).to(param.device)

                if param.shape == w_tensor.shape:
                    new_state_dict[name] = w_tensor
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {w_tensor.shape}. "
                                f"Skipping this layer.")
                    new_state_dict[name] = param  # Retain original
            else:
                logger.warning(f"Parameter for {name} is not a NumPy array. Keeping original value.")
                new_state_dict[name] = param

        # Load updated state_dict with strict=False to allow partial updates
        model.model.load_state_dict(new_state_dict)
        missing, unexpected = model.model.load_state_dict(new_state_dict, strict=False)
        logger.debug(f"Updated layers: {len(new_state_dict) - len(missing)} | Missing layers: {len(missing)} | Unexpected layers: {len(unexpected)}")

        logger.debug("Model weights updated successfully.")
        return True
    except Exception as e:
        logger.error(f"set_weights error: {e}", exc_info=True)
        return False


# Function to load YOLOv8 model with .pt weights
def load_yolo_model(weight_path):
    """
    Load a YOLOv8 model from a .pt file.
    :param weight_path: Path to the .pt weight file.
    :return:            Loaded YOLOv8 model.
    """
    try:
        logger.debug(f"Loading YOLOv8 model from {weight_path}...")
        model = YOLO(weight_path)
        model = load_yolo_model("yolov8.pt")
        if model is None:
            raise RuntimeError("Failed to load YOLOv8 model.")

        logger.debug("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}", exc_info=True)
        return None
