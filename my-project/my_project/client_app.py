import logging
import os
import warnings
import torch
import platform
from flwr.common import Code, Status
from flwr.client import ClientApp, Client
from flwr.common import Context, FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters, RecordSet

from ultralytics import YOLO
from my_project.task import (
    download_model, 
    get_data_yaml_path, 
    update_data_yaml_paths, 
    IS_WINDOWS, 
    OS_NAME
)  # Import OS detection
import urllib
from my_project.get_set_model import get_weights, load_yolo_model, set_weights
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("client", "logs/client.log")

# Log OS detection information
logger.info(f"[Client] Detected operating system: {OS_NAME}, IS_WINDOWS={IS_WINDOWS}")

# Constants
DEFAULT_BATCH_ID_RANGE = (1, 10)
DEFAULT_IMAGE_SIZE = 640

class FlowerClient(Client):

    def __init__(self,
                model_path: str,
                client_state : RecordSet,
                local_epochs : int,
                batch_id_range: tuple = DEFAULT_BATCH_ID_RANGE,  # Using constant
                ):
    
        super().__init__()
        self.model_path = model_path
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.batch_id_range = batch_id_range
        self.batch_id = None
        
        # Decide GPU/CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Client] Initializing FlowerClient with model={model_path} on {self.device}")
        # Load YOLO model with valid initial weights
        try:
            if not os.path.exists(self.model_path):
                logger.warning("[Client] Model weights not found. Downloading default YOLOv8 weights.")
                download_model()  # Ensure we have some valid weights to avoid shape mismatch
            
            self.yolo = YOLO(self.model_path)
            self.model = self.yolo.model
            
            # Set number of classes if needed
            self.model.nc = 13
            if hasattr(self.model, 'head'):
                self.model.head.nc = 13
                
            logger.info("[Client] YOLO model loaded successfully.")
        except Exception as e:
            logger.error("[Client] Failed to load YOLO model!", exc_info=True)
            self.model = None
            self.yolo = None
        
    def _validate_batch_id(self, batch_id: int) -> bool:
        """Validate that batch_id is within the acceptable range."""
        if not isinstance(batch_id, int):
            logger.warning(f"[Client] Batch ID must be an integer. Got {type(batch_id)}.")
            return False
        if batch_id < self.batch_id_range[0] or batch_id > self.batch_id_range[1]:
            logger.warning(f"[Client] Batch ID {batch_id} is out of range {self.batch_id_range}.")
            return False
        return True    

    def _get_data_yaml_path(self, batch_id):
        """
        Gets the data.yaml path and ensures it's updated for the current OS.
        
        Args:
            batch_id: The batch ID
            
        Returns:
            Path: Path to the data.yaml file
        """
        # Get the path
        yaml_path = get_data_yaml_path(batch_id)
        
        # Update the path in data.yaml if needed
        if os.path.exists(yaml_path):
            update_data_yaml_paths(batch_id)
            
        return yaml_path

    def fit(self, ins: FitIns) -> FitRes:
        if self.model is None:
            logger.error("[Client] No model available, cannot train.")
            # Return original parameters so server isn't disrupted
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to load model for training"),
            )
            
        # 1) Load weights from parameters 
        weights_list = self._parameters_to_list(ins.parameters)
        if not weights_list:
            logger.warning("[Client] Received empty weights. Using current model weights.")
            weights_list = get_weights(self.model)
        
        # Track weights checksum for debugging weight transfer
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)
        logger.info(f"[Client] Received weights with checksum: {weights_checksum}")
        
        try:
            set_weights(self.model, weights_list)
            logger.info("[Client] Successfully applied received weights to model")
        except Exception as e:
            logger.error(f"[Client] set_weights failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to load weights for training"),
            )

        # 2) Parse config from server
        batch_id = ins.config.get("batch_id", None)
        local_epochs = ins.config.get("local_epochs", self.local_epochs)

        # Use stored batch_id as fallback if available
        if batch_id is None:
            logger.warning("[Client] No batch_id received, using stored batch_id.")
            batch_id = self.batch_id

        # Validate batch_id
        if batch_id is None or not self._validate_batch_id(batch_id):
            logger.error(f"[Client] Invalid batch_id: {batch_id}. Expected range: {self.batch_id_range}.")
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message=f"Invalid batch_id: {batch_id}"),
            )

        # Store valid batch_id for future use
        self.batch_id = batch_id
        logger.info(f"[Client] Starting local training with batch_id={self.batch_id}, local_epochs={local_epochs}")

        # 3) Build data path and verify it exists
        data_yaml_path = str(self._get_data_yaml_path(self.batch_id))
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Training data.yaml not found: {data_yaml_path}")
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing training data.yaml"),
            )

        # 4) Train the model
        try:
            logger.info(f"[Client] Training with data config: {data_yaml_path}")
            results = self.yolo.train(
                data=data_yaml_path,
                epochs=local_epochs,
                imgsz=DEFAULT_IMAGE_SIZE,
                device=self.device,
                verbose=False
            )
            
            # 5) Process results
            if hasattr(results, "results_dict"):
                num_examples = 10  # Default value if exact count is not available
                results_dict = results.results_dict
                
                # Create metrics dictionary
                metrics = {
                    "precision": results_dict.get("metrics/precision(B)", 0),
                    "recall": results_dict.get("metrics/recall(B)", 0),
                    "mAP50": results_dict.get("metrics/mAP50(B)", 0),
                    "mAP50-95": results_dict.get("metrics/mAP50-95(B)", 0),
                    "fitness": results_dict.get("fitness", 0),
                    "os": OS_NAME  # Include OS information in metrics for tracking
                }
                
                logger.info(f"[Client] {self.batch_id} Training done. metrics={metrics}")
            else:
                logger.warning("[Client] Training did not return metrics.")
                num_examples = 0
                metrics = {"error": "No metrics returned from training", "os": OS_NAME}

            # 6) Extract updated weights for sending back to server
            updated_weights = get_weights(self.model)
            updated_checksum = sum(w.sum() for w in updated_weights if w.size > 0)
            logger.info(f"[Client] Sending back weights with checksum: {updated_checksum}")
            
            return FitRes(
                parameters=self._list_to_parameters(updated_weights),
                num_examples=int(num_examples),
                metrics=metrics,
                status=Status(code=Code.OK, message="Training successful"),
            )
        except Exception as e:
            logger.error(f"[Client] YOLO training failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"error": str(e), "os": OS_NAME},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training process failed"),
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.model is None:
            logger.error("[Client] No model available, cannot evaluate.")
            return EvaluateRes(
                loss=float("inf"), 
                num_examples=1, 
                metrics={"os": OS_NAME}, 
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No model available")
            )
            
        # 1) Load weights from parameters
        weights_list = self._parameters_to_list(ins.parameters)
        if not weights_list:
            logger.warning("[Client] Received empty weights for evaluation. Using current model weights.")
            weights_list = get_weights(self.model)
            
        # Track weights checksum for debugging weight transfer
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)
        logger.info(f"[Client] Received evaluation weights with checksum: {weights_checksum}")
            
        # 2) Apply weights to model
        try:
            logger.info("[Client] Setting evaluation weights.")
            success = set_weights(self.model, weights_list)
            if not success:
                raise ValueError("set_weights returned False - mismatch or error.")
        except Exception as e:
            logger.error(f"[Client] evaluate() set_weights failed: {e}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Failed to load weights for evaluation"),
            )

        # 3) Parse config from server
        batch_id = ins.config.get("batch_id", None)
        
        # Use stored batch_id as fallback if available
        if batch_id is None:
            logger.warning("[Client] No batch_id received for evaluation, using stored batch_id.")
            batch_id = self.batch_id
            
        # Validate batch_id
        if batch_id is None or not self._validate_batch_id(batch_id):
            logger.error(f"[Client] Invalid batch_id for evaluation: {batch_id}. Expected range: {self.batch_id_range}.")
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message=f"Invalid batch_id: {batch_id}"),
            )
            
        # Store valid batch_id for future use
        self.batch_id = batch_id
        logger.info(f"[Client] Starting evaluation with batch_id={self.batch_id}")

        # 4) Build data path and verify it exists
        data_yaml_path = str(self._get_data_yaml_path(self.batch_id))
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Evaluation data.yaml not found: {data_yaml_path}")
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Missing evaluation data.yaml"),
            )

        # 5) Evaluate the model
        try:
            logger.info(f"[Client] Evaluating with data config: {data_yaml_path}")
            results = self.yolo.val(
                data=data_yaml_path,
                imgsz=DEFAULT_IMAGE_SIZE,
                device=self.device,
                verbose=False
            )
            
            # 6) Process results
            num_examples = 10  # Default value if exact count is not available
            fitness_value = results.results_dict.get("fitness", float("inf"))
            
            # Create metrics dictionary
            metrics = {
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "os": OS_NAME  # Include OS information in metrics for tracking
            }

            logger.info(f"[Client] Evaluation done. Loss={fitness_value}, metrics={metrics}, Images Processed={num_examples}")

            return EvaluateRes(
                loss=float(fitness_value),
                num_examples=num_examples,
                metrics=metrics,
                status=Status(code=Code.OK, message="Evaluation successful"),
            )
        except Exception as e:
            error_message = f"YOLO evaluation failed due to: {str(e)}"
            logger.error(f"[Client] {error_message}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"error": error_message, "os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message=error_message),
            )

    def _list_to_parameters(self, weights_list) -> Parameters:
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(weights_list)

    def _parameters_to_list(self, parameters: Parameters):
        from flwr.common import parameters_to_ndarrays
        return parameters_to_ndarrays(parameters)


def client_fn(context: Context):
    try:
        logger.info("[Client] Creating FlowerClient instance from client_fn.")
        model_path = "models/yolov8s.pt"  # Relative path

        local_epochs = context.run_config.get("local_epochs", 1)
        client_state = context.state
        batch_id_range = context.run_config.get("batch_id_range", DEFAULT_BATCH_ID_RANGE)
        
        client = FlowerClient(
            model_path=model_path,
            client_state=client_state,
            local_epochs=local_epochs,
            batch_id_range=batch_id_range,
        )
        
        return client
    except Exception as e:
        logger.error(f"[Client] client_fn failed to create FlowerClient: {e}", exc_info=True)
        return None


# So you can start the client with: flwr run --app client:app
app = ClientApp(client_fn=client_fn)
