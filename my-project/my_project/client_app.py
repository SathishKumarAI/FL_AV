import logging
import os
import warnings
import torch
from flwr.common import Code, Status
from flwr.client import ClientApp, Client
from flwr.common import Context, FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters

from ultralytics import YOLO
from my_project.task import get_weights, set_weights  # Custom YOLO utility functions
import urllib
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("client", "logs/client.log")

# Model Path & Download URL
MODEL_PATH = "models/yolov5su.pt"
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5su.pt"


def download_model():
    """Download YOLO model if not found."""
    logger.info(f"[Client] Model not found. Downloading from {MODEL_URL}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure directory exists
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("[Client] Model downloaded successfully.")
    except Exception as e:
        logger.error(f"[Client] Failed to download the model: {e}", exc_info=True)
        raise RuntimeError("Client cannot start without a valid YOLO model.") from e

class FlowerClient(Client):
    """
    A low-level Flower Client that uses YOLOv5 for training/validation.
    It receives `batch_id` and other config from the server's FitIns.config/EvaluateIns.config,
    thereby referencing a unique data.yaml each round.
    """

    def __init__(self, model_path: str = "models/yolov5su.pt"):
        """
        :param model_path: Path to your YOLOv5 .pt weights (e.g., 'models/yolov5su.pt').
        """
        super().__init__()
        self.model_path = model_path

        # Decide GPU/CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Client] Initializing FlowerClient with model={model_path} on {self.device}")
        try:
            if not os.path.exists(self.model_path):
                download_model()
            self.model = YOLO(self.model_path).to(self.device)
            logger.info("[Client] YOLO model loaded successfully.")
        except Exception as e:
            logger.error("[Client] Failed to load YOLO model!", exc_info=True)
            self.model = None
            
    def fit(self, ins: FitIns) -> FitRes:
        """
        Called by the Flower server to perform local training.
        Steps:
        1) Extract global weights from ins.parameters and load into YOLO model.
        2) Parse config (batch_id, local_epochs).
        3) Train the model on local data.yaml (constructed from batch_id).
        4) Return updated model weights + num_examples + metrics.
        """
        if self.model is None:
            logger.error("[Client] No model available, cannot train.")
            # Return original parameters so server isn't disrupted
            return FitRes(
                parameters=ins.parameters,
                num_examples=0,
                metrics={"train_loss": float("inf")}
            )

        # 1) Convert parameters to local YOLO weights
        try:
            logger.info("[Client] Loading global weights into local YOLO model.")
            success = set_weights(self.model, self._parameters_to_list(ins.parameters))
            if not success:
                raise ValueError("set_weights returned False - shape mismatch or error.")
        except Exception as e:
            logger.error(f"[Client] fit() set_weights failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=0,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to set load model weights"),
            )
        # 2) Parse config (no .get usage, so we do direct indexing)
        config = ins.config
        try:
            batch_id = config["batch_id"]        # If missing, KeyError
            local_epochs = config["local_epochs"] # If missing, KeyError
            logger.info(
                f"[Client] Starting local training with batch_id={batch_id}, local_epochs={local_epochs}"
            )
        except KeyError as ke:
            logger.error(f"[Client] Missing key in FitIns.config: {ke}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=0,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to load model weights"),
            )

        # Extract training config
        config = ins.config
        try:
            batch_id = config["batch_id"]
            local_epochs = config["local_epochs"]
            logger.info(f"[Client] Starting training with batch_id={batch_id}, local_epochs={local_epochs}")
        except KeyError as ke:
            logger.error(f"[Client] Missing key in FitIns.config: {ke}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=0,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing batch_id in training config"),
            )

        # Construct training data path
        data_yaml_path = f"batch/batch_{batch_id}/data.yaml"
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Training data.yaml not found: {data_yaml_path}")
            return FitRes(
                parameters=ins.parameters,
                num_examples=0,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing training data.yaml"),
            )

        try:
            results = self.model.train(
                data=data_yaml_path,
                epochs=local_epochs,
                imgsz=640,
                device=self.device,
                verbose=False
            )
            final_loss = results.results_dict.get("train/loss", float("inf"))
            num_examples = results.results_dict.get("training/images", 0)

            logger.info(
                f"[Client] Local training done. final_loss={final_loss}, images={num_examples}"
            )
            
            # 4) Extract updated weights
            # updated_weights = get_weights(self.model)
            # Convert them back to Flower Parameters
            # updated_params = self._list_to_parameters(updated_weights)

            # Provide some metrics
            # metrics = {"train_loss": final_loss}
            return FitRes(
                parameters=self._list_to_parameters(get_weights(self.model)),  # Updated weights
                num_examples=int(num_examples),
                metrics={"train_loss": final_loss},
                status=Status(code=Code.OK, message="Training successful"),
            )
        except Exception as e:
            logger.error(f"[Client] YOLO training failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=0,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training process failed"),
            )

        # return FitRes(parameters=updated_params, num_examples=int(num_examples), metrics=metrics, status=Status(code=Code.OK, message="Returning fallback after error"))

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Called by the Flower server to perform local evaluation.
        Steps:
        1) Load global weights into local YOLO model.
        2) Parse config (batch_id).
        3) Evaluate on local data.yaml.
        4) Return loss, num_examples, and metrics.
        """
        if self.model is None:
            logger.error("[Client] No model available, cannot evaluate.")
            return EvaluateRes(loss=float("inf"), num_examples=0, metrics={})

        # 1) Load global YOLO weights
        try:
            logger.info("[Client] Setting evaluation weights.")
            success = set_weights(self.model, self._parameters_to_list(ins.parameters))
            if not success:
                raise ValueError("set_weights returned False - mismatch or error.")
        except Exception as e:
            logger.error(f"[Client] evaluate() set_weights failed: {e}", exc_info=True)
            return EvaluateRes(
            loss=float("inf"),
            num_examples=0,
            metrics={},
            status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED , message="Failed to load weights for evaluation"),
        )

        # 2) Parse config
        config = ins.config
        try:
            batch_id = config.get("batch_id", None)  # Use .get() to avoid KeyError
            if batch_id is None:
                raise ValueError("[Client] Server did not send batch_id for evaluation.")
            logger.info(f"[Client] Local evaluation with batch_id={batch_id}")
        except Exception as e:
            logger.error(f"[Client] Evaluation config missing required key: {e}", exc_info=True)
            return EvaluateRes(
            loss=float("inf"),
            num_examples=0,
            metrics={},
            status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED , message="Missing batch_id in evaluation config"),
        )

        # 3) Evaluate on the data.yaml
        data_yaml_path = f"batch/batch_{batch_id}/data.yaml"
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Evaluation data.yaml not found: {data_yaml_path}")
            return EvaluateRes(
                loss=float("inf"),
                num_examples=0,
                metrics={},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED , message="Missing evaluation data.yaml"),
            )

        try:
            results = self.model.val(
                data=data_yaml_path,
                imgsz=640,
                device=self.device,
                verbose=False
            )
            val_loss = results.results_dict.get("val/loss", float("inf"))
            mAP50 = results.results_dict.get("metrics/mAP50", 0.0)
            num_examples = results.results_dict.get("validation/images", 0)
            metrics = {"mAP50": mAP50}

            logger.info(f"[Client] Evaluation done. Loss={val_loss}, mAP50={mAP50}, Images Processed={num_examples}")

            return EvaluateRes(
                loss=float(val_loss),
                num_examples=int(num_examples),
                metrics={"mAP50": mAP50},
                status=Status(code=Code.OK, message="Evaluation successful"),
            )
        except Exception as e:
            logger.error(f"[Client] YOLO evaluation failed: {e}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=0,
                metrics={},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED , message="Evaluation process failed"),
            )
    # ----------------------------------------------------------------
    # Helper conversions: Parameters <-> list of numpy arrays
    # If you use the same approach as "NumPyClient", you can skip these.
    # But the low-level Client requires manual conversion.
    # ----------------------------------------------------------------
    def _list_to_parameters(self, weights_list) -> Parameters:
        """
        Convert a list of NumPy arrays into Flower Parameters.
        Equivalent to fl.common.ndarrays_to_parameters, but shown here if you want inline code.
        """
        from flwr.common import NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
        import numpy as np

        # Each item in weights_list is a np.ndarray
        # We simply call the utility if you want:
        return ndarrays_to_parameters(weights_list)

    def _parameters_to_list(self, parameters: Parameters):
        """
        Convert Flower Parameters into a list of NumPy arrays.
        Equivalent to fl.common.parameters_to_ndarrays.
        """
        from flwr.common import parameters_to_ndarrays
        return parameters_to_ndarrays(parameters)


def client_fn(context: Context):
    """
    The function that Flower calls to create a new client instance.
    This is used if you run 'flwr run --app client:app'.
    """
    try:
        logger.info("[Client] Creating FlowerClient instance from client_fn.")
        model_path = "models/yolov5su.pt"  # Or dynamically read from context if desired
        client = FlowerClient(model_path=model_path)
        return client
    except Exception as e:
        logger.error("[Client] client_fn failed to create FlowerClient", exc_info=True)
        return None


# So you can start the client with: flwr run --app client:app
app = ClientApp(client_fn=client_fn)
