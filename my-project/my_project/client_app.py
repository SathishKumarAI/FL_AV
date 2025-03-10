import logging
import os
import warnings
import torch
from flwr.common import Code, Status
from flwr.client import ClientApp, Client
from flwr.common import Context, FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters, RecordSet

from ultralytics import YOLO
from my_project.task import  download_model  # Custom YOLO utility functions
import urllib
from my_project.get_set_model import get_weights, load_yolo_model, set_weights
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("client", "logs/client.log")


class FlowerClient(Client):

    def __init__(self,
                model_path: str ,
                # = "models/yolov8s.pt",
                client_state : RecordSet,
                local_epochs : int,
                batch_id_range: tuple = (1, 10),  # Default range for batch IDs
                ):
    
        super().__init__()
        self.model_path = model_path
        # self.yolo = YOLO(self.model_path)
        # self.model = self.yolo.model
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.batch_id_range = batch_id_range
        
        # Decide GPU/CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Client] Initializing FlowerClient with model={model_path} on {self.device}")
        # Load YOLO model with valid initial weights
        try:
            if not os.path.exists(self.model_path):
                logger.warning("[Client] Model weights not found. Downloading default YOLOv8 weights.")
                download_model()  # Ensure we have some valid weights to avoid shape mismatch
            # self.model = load_yolo_model(self.model_path).to(self.device)
            # self.model = load_yolo_model(self.model_path)
            self.yolo = YOLO(self.model_path)
            self.model = self.yolo.model
            # Set number of classes if needed
            self.model.nc = 13
            if hasattr(self.model, 'head'):
                self.model.head.nc = 13
            logger.info("[Client] YOLO model loaded successfully.")
            logger.info("[Client] YOLO model loaded successfully.")
        except Exception as e:
            logger.error("[Client] Failed to load YOLO model!", exc_info=True)
            self.model = None
            self.yolo = None

        # self.model = model
        
    def _validate_batch_id(self, batch_id: int) -> bool:
        """Validate that batch_id is within the acceptable range."""
        if not isinstance(batch_id, int):
            logger.warning(f"[Client] Batch ID must be an integer. Got {type(batch_id)}.")
            return False
        if batch_id < self.batch_id_range[0] or batch_id > self.batch_id_range[1]:
            logger.warning(f"[Client] Batch ID {batch_id} is out of range {self.batch_id_range}.")
            return False
        return True    

    def fit(self, ins: FitIns ) -> FitRes:

        if self.model is None:
            logger.error("[Client] No model available, cannot train.")
            # Return original parameters so server isn't disrupted
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Failed to load model for evaluation"),
            )
            
        # 1️⃣ Ensure Weights Match YOLOv8 Structure
        weights_list = self._parameters_to_list(ins.parameters)
        if not weights_list:
            logger.warning("[Client] Received empty weights. Using default YOLOv8 pretrained weights.")
            weights_list = get_weights(self.model)  # Use YOLOv8 default weights if first iteration
        # logger.info(f"[Client]  weights_list={weights_list}")
        # # 1) Convert parameters to local YOLO weights
        
        try:
            set_weights(self.model, weights_list)
        except Exception as e:
            logger.error(f"[Client] set_weights failed: {e}", exc_info=True)
            return  FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Failed to load model for evaluation"),
            )

        # 2) Parse config (no .get usage, so we do direct indexing)
        config = ins.config
        # batch_id = config.get("batch_id", None)
        batch_id = ins.config.get("batch_id", None)

        # Use stored batch_id as fallback if available
        if batch_id is None:
            logger.warning("[Client] No batch_id received, using stored batch_id.")
            batch_id = getattr(self, "batch_id", None)

        # Validate batch_id
        if batch_id is None or not self._validate_batch_id(batch_id):
            logger.error(f"[Client] Invalid batch_id: {batch_id}. Expected range: {self.batch_id_range}.")
            return FitRes(  # or EvaluateRes for evaluate method
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message=f"Invalid batch_id: {batch_id}"),
            )

        # Store valid batch_id for future use
        self.batch_id = batch_id

        try:
            self.batch_id = config["batch_id"]        # If missing, KeyError
            local_epochs = config["local_epochs"] # If missing, KeyError
            logger.info(
                f"[Client] Starting local training with batch_id={self.batch_id}, local_epochs={local_epochs}"
            )
        except KeyError as ke:
            logger.error(f"[Client] Missing key in FitIns.config: {ke}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing batch_id or local_epochs in training config"),
            )


        # Construct training data path
        data_yaml_path = f"batch/batch_{self.batch_id}/data.yaml"
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Training data.yaml not found: {data_yaml_path}")
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing training data.yaml"),
            )

        try:
            results = self.yolo.train(
                data=data_yaml_path,
                epochs=local_epochs,
                imgsz=640,
                device=self.device,
                verbose=False
            )

            
            # Ensure training completed
            if hasattr(results, "results_dict"):

                # num_examples = results.results_dict.get("training/images", 1)
                num_examples = 10
                # Assuming 'results' is your DetMetrics object
                # # First convert the DetMetrics object to a dictionary
                results_dict = results.results_dict  # This is the correct way to access the dictionary

                fitness_value = results_dict["fitness"]  # Access from the dictionary
                # Create metrics dictionary
                metrics = {
                    # "train_loss": fitness_value,
                    "precision": results.results_dict.get("metrics/precision(B)", 0),
                    "recall": results.results_dict.get("metrics/recall(B)", 0),
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0)
                }
                # dir1 = dir(results)
                # dir = ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 
                # 'ap_class_index', 
                # 'box', 
                # 'class_result', 
                # 'confusion_matrix', 
                # 'curves', 
                # 'curves_results', 
                # 'fitness', 'keys', 'maps', 
                # 'mean_results', 'names', 'on_plot', 'plot', 'process', 
                # 'results_dict', 
                # 'save_dir', 'speed', 'task']
                
                logger.info(f"[Client] {self.batch_id}  Training done. metrics = {metrics}    results_dict = {results_dict}   "     )
                # logger.info(f"[Client]{self.batch_id} Training done. Loss={fitness_value}, Images={num_examples}  results ={results}"           )
            else:
                logger.warning("[Client] Training did not return metrics.")
                final_loss = float("inf")
                num_examples = 0

            return FitRes(
                parameters=self._list_to_parameters(get_weights(self.model)),
                num_examples=int(num_examples),
                metrics=metrics,
                status=Status(code=Code.OK, message="Training successful"),
            )
        except Exception as e:
            logger.error(f"[Client] YOLO training failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training process failed"),
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.model is None:
            logger.error("[Client] No model available, cannot evaluate.")
            return EvaluateRes(loss=float("inf"), num_examples=1, metrics={}, status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No model available"),)
        #1️⃣ Load Weights (Ensure They Match)
        weights_list = self._parameters_to_list(ins.parameters)
        if not weights_list:
            if self.model is None:
                logger.error("[Client] No model and no weights received, cannot evaluate.")
                return EvaluateRes(
                    loss=float("inf"),
                    num_examples=1,
                    metrics={},
                    status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No model weights available"),
                )
            logger.warning("[Client] Received empty weights. Using current YOLO model weights.")
            weights_list = get_weights(self.model)
            
        # 1) Load global YOLO weights
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
            metrics={},
            status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED , message="Failed to load weights for evaluation"),
        )

        # 2) Parse config\\
        
        # config = ins.config.get()
        
        batch_id = ins.config.get("batch_id", None)

        # Store valid batch_id for future use
        self.batch_id = batch_id

        try:
            self.batch_id = ins.config["batch_id"]        # If missing, KeyError
            local_epochs = ins.config["local_epochs"] # If missing, KeyError
            logger.info(
                f"[Client] Starting local training with batch_id={self.batch_id}, local_epochs={local_epochs}"
            )
        except KeyError as ke:
            logger.error(f"[Client] Missing key in FitIns.config: {ke}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing batch_id or local_epochs in training config"),
            )


        # 3) Evaluate on the data.yaml
        data_yaml_path = f"batch/batch_{batch_id}/data.yaml"
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Evaluation data.yaml not found: {data_yaml_path}")
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED , message="Missing evaluation data.yaml"),
            )

        try:
            results = self.yolo.val(
                data=data_yaml_path,
                imgsz=640,
                device=self.device,
                verbose=False
            )
            # val_loss = results.results_dict.get("val/loss", float("inf"))
            # mAP50 = results.results_dict.get("metrics/mAP50", 0.0)
            # num_examples = results.results_dict.get("validation/images", 0)
            
            num_examples = 10
            fitness_value = results.results_dict["fitness"]  # Access from the dictionary
            # Create metrics dictionary
            metrics = {
                # "train_loss": fitness_value,
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0)
            }

            logger.info(f"[Client] Evaluation done. Loss = {fitness_value}, metrics = {metrics}, Images Processed = {num_examples} ")

            return EvaluateRes(
                loss=float(fitness_value),
                num_examples=1,
                metrics= metrics,
                status=Status(code=Code.OK, message="Evaluation successful"),
            )
        except Exception as e:
            error_message = f"YOLO evaluation failed due to: {str(e)}"
            logger.error(f"[Client] {error_message}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"error": error_message},  # Return error message in metrics
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message=error_message),
            )

    def _list_to_parameters(self, weights_list) -> Parameters:
        from flwr.common import NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
        import numpy as np

        return ndarrays_to_parameters(weights_list)

    def _parameters_to_list(self, parameters: Parameters):
        from flwr.common import parameters_to_ndarrays
        return parameters_to_ndarrays(parameters)


def client_fn(context: Context):

    try:
        logger.info("[Client] Creating FlowerClient instance from client_fn.")
        model_path = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov8s.pt"  # Or dynamically read from context if desired

        local_epochs = context.run_config["local_epochs"]
        client_state = context.state
        batch_id_range = context.run_config.get("batch_id_range", (1, 10))  # Default range if not provided
        client = FlowerClient(
            model_path = model_path,
            client_state = client_state,
            local_epochs = local_epochs,
            batch_id_range=batch_id_range,
                            )
        
        return client
    # .to_client()
    except Exception as e:
        logger.error("[Client] client_fn failed to create FlowerClient", exc_info=True)
        return None


# So you can start the client with: flwr run --app client:app
app = ClientApp(client_fn=client_fn)
