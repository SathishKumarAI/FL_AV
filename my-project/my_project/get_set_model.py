from typing import List
import warnings
import torch
import numpy as np
import platform
from collections import OrderedDict
# NOTE: this module deliberately does not import ultralytics. The weight-transfer
# helpers (get_weights/set_weights) depend only on torch + numpy, which keeps FedAvg
# serialization testable and importable without the heavy dependency.
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("get_set", "logs/get_set.log")

# OS detection
IS_WINDOWS = platform.system() == "Windows"
OS_NAME = platform.system()
logger.info(f"[GetSet] Detected operating system: {OS_NAME}")

# Constants
DEFAULT_NUM_CLASSES = 13  # The default number of classes for our model

# The architecture both server and client must build, so their state_dicts match.
# Ultralytics infers the scale from the *filename*: "yolov8s-13" parses as scale
# 's', matching models/yolov8s.pt. The old models/yolo8n.yaml parsed as 'n' and
# would silently build a nano net to load small weights into.
NUM_CLASSES_MODEL_YAML = "models/yolov8s-13.yaml"

def get_normalized_path(path):
    """
    Normalize path to the current operating system format.
    
    Args:
        path: Path to normalize
        
    Returns:
        str: Normalized path for the current OS
    """
    if IS_WINDOWS:
        return str(path).replace('/', '\\')
    else:
        return str(path).replace('\\', '/')

def get_weights(model):
    """
    Extract the FULL model state as a list of NumPy arrays, in ``state_dict`` order.

    Unlike ``model.parameters()`` (learnable tensors only), ``state_dict()`` also
    includes registered buffers — crucially the BatchNorm running statistics
    (``running_mean``, ``running_var``, ``num_batches_tracked``). YOLOv8 is
    BatchNorm-heavy, so transferring only parameters means FedAvg never averages
    those statistics and the federated model is incorrect. We therefore serialize
    the whole ``state_dict``.

    Order: ``state_dict()`` preserves a deterministic insertion order for a fixed
    module structure, so the i-th array here corresponds to the i-th key of
    ``model.state_dict().keys()`` on any process that builds the same architecture.
    Only the ordered arrays travel over the wire; keys are reconstructed locally in
    ``set_weights`` (see its invariant note).

    Args:
        model: A loaded YOLOv8 model instance (DetectionModel / nn.Module).

    Returns:
        A list of NumPy arrays (one per state_dict entry), or empty list on error.
    """
    try:
        logger.debug("[GetSet] Extracting YOLOv8 model state_dict (params + buffers)...")
        weights_list = [t.detach().cpu().numpy() for t in model.state_dict().values()]

        # Calculate checksum for debugging weight transport.
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)

        logger.debug(
            f"[GetSet] Extracted {len(weights_list)} state tensors "
            f"(params+buffers) with checksum: {weights_checksum}"
        )
        return weights_list
    except Exception as e:
        logger.error(f"[GetSet] get_weights error: {e}", exc_info=True)
        return []


def set_weights(model, parameters: List[np.ndarray]) -> bool:
    """
    Apply a full ``state_dict`` (params + buffers) received as ordered NumPy arrays.

    The incoming list is zipped, in order, against ``model.state_dict().keys()``
    recomputed locally, rebuilt into an ``OrderedDict``, and loaded with
    ``load_state_dict(..., strict=True)``.

    **Invariant:** both the server and every client must construct the *identical*
    architecture (``YOLO("models/yolov8s.pt").model`` with ``nc=13``) before calling
    this. Then the local key order matches the order ``get_weights`` used on the
    sender, so positional zip is correct. ``strict=True`` raises loudly if the order
    or set of keys ever drifts — the desired failure mode, not a silent partial load.

    Each integer buffer (notably BatchNorm ``num_batches_tracked``, int64) is cast
    back to the model's own dtype/device for that key, so ``load_state_dict``'s dtype
    check passes (a raw ``from_numpy`` of an int array would otherwise mismatch).

    Args:
        model: The YOLOv8 model instance.
        parameters: Ordered list of NumPy arrays (full state_dict).

    Returns:
        bool: True on success, False on error.
    """
    try:
        logger.debug("[GetSet] Setting YOLOv8 model state_dict (params + buffers)...")

        # Calculate checksum for debugging.
        weights_checksum = sum(w.sum() for w in parameters if w.size > 0)
        logger.debug(f"[GetSet] Applying state with checksum: {weights_checksum}")

        ref_state = model.state_dict()
        keys = list(ref_state.keys())
        if len(parameters) != len(keys):
            logger.error(
                f"[GetSet] State count mismatch: model has {len(keys)} state tensors, "
                f"but received {len(parameters)} arrays. "
                f"(Are server and client building the same architecture?)"
            )
            return False

        new_state = OrderedDict()
        for key, arr in zip(keys, parameters):
            ref = ref_state[key]
            tensor = torch.as_tensor(arr).to(dtype=ref.dtype, device=ref.device)
            if tensor.shape != ref.shape:
                logger.error(
                    f"[GetSet] Shape mismatch at '{key}': "
                    f"model {tuple(ref.shape)} != received {tuple(tensor.shape)}"
                )
                return False
            new_state[key] = tensor

        model.load_state_dict(new_state, strict=True)
        logger.debug("[GetSet] Model state_dict updated successfully.")
        return True
    except Exception as e:
        logger.error(f"[GetSet] Error in set_weights: {e}", exc_info=True)
        return False
