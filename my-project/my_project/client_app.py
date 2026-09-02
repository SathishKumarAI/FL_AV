import logging
import math
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
    materialize_data_yaml,
    count_shard_examples,
    get_optimal_batch_size,
    IS_WINDOWS,
    OS_NAME
)  # Import OS detection
import urllib
from my_project.get_set_model import (NUM_CLASSES_MODEL_YAML, get_weights, set_weights,
                                      warm_start_head)
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("client", "logs/client.log")

# Log OS detection information
logger.info(f"[Client] Detected operating system: {OS_NAME}, IS_WINDOWS={IS_WINDOWS}")

# Constants
DEFAULT_BATCH_ID_RANGE = (1, 10)
# Ultralytics' nominal batch size: it accumulates gradients up to this many
# images before stepping the optimizer (ultralytics default `nbs`).
NOMINAL_BATCH_SIZE = 64
DEFAULT_IMAGE_SIZE = 640

class FlowerClient(Client):

    def __init__(self,
                model_path: str,
                client_state : RecordSet,
                local_epochs : int,
                batch_id_range: tuple = DEFAULT_BATCH_ID_RANGE,  # Using constant
                cache: str = "",
                ):

        super().__init__()
        self.model_path = model_path
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.batch_id_range = batch_id_range
        self.batch_id = None
        # "" | "ram" | "disk". Ultralytics wants False, not "", for "no cache".
        self.cache = cache or False
        
        # Decide GPU/CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Client] Initializing FlowerClient with model={model_path} on {self.device}")
        # Load YOLO model with valid initial weights
        try:
            if not os.path.exists(self.model_path):
                logger.warning("[Client] Model weights not found. Downloading default YOLOv8 weights.")
                download_model()  # Ensure we have some valid weights to avoid shape mismatch
            
            # Build the 13-class architecture from the yaml, then load the COCO
            # weights into it. Constructing from the .pt alone gives an 80-class
            # head; `model.nc = 13` only renames an attribute, it does not rebuild
            # the head, so the server and client end up with different archs the
            # moment Ultralytics rebuilds the head from data.yaml during train().
            # The server builds the same arch (server_app.py) — set_weights'
            # strict=True is what enforces that they agree.
            self.yolo = YOLO(NUM_CLASSES_MODEL_YAML).load(self.model_path)
            # The head the server broadcasts overwrites this one every round, so this
            # matters only for the shape of the model before the first set_weights.
            # Done anyway, and through the same helper, so the two sides cannot drift:
            # a client whose head is warmed differently from the server's is exactly
            # the kind of asymmetry set_weights' strict=True exists to catch.
            warm_start_head(self.yolo.model, YOLO(self.model_path))

            logger.info("[Client] YOLO model loaded successfully.")
        except Exception as e:
            logger.error("[Client] Failed to load YOLO model!", exc_info=True)
            self.yolo = None

    @property
    def model(self):
        """The live DetectionModel.

        Ultralytics rebinds ``yolo.model`` to the trained checkpoint at the end of
        ``train()``. Caching it in ``__init__`` left an orphan: the client returned
        the weights it was *sent*, never the ones it *trained*, so FedAvg averaged
        identical inputs and the federation learned nothing. Reading through means
        every call site sees the current module, including ones added later.
        """
        return self.yolo.model if self.yolo is not None else None


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
        Return a runtime data.yaml with correct absolute paths for this machine.

        Uses materialize_data_yaml(), which writes a gitignored data.runtime.yaml
        instead of mutating the tracked data.yaml.

        Args:
            batch_id: The batch ID

        Returns:
            Path: Path to the runtime data.yaml (falls back to the tracked one).
        """
        if os.path.exists(get_data_yaml_path(batch_id)):
            return materialize_data_yaml(batch_id)
        return get_data_yaml_path(batch_id)

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
            # set_weights returns False on a count/shape mismatch instead of raising.
            # Ignoring it made a failed load look like a successful round: the client
            # trained from its own stale weights and the server averaged them in.
            if not set_weights(self.model, weights_list):
                raise ValueError("set_weights returned False - mismatch or error.")
            logger.info("[Client] Successfully applied received weights to model")
        except Exception as e:
            logger.error(f"[Client] set_weights failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to load weights for training"),
            )

        # Snapshot the global weights so we can apply a FedProx proximal pull
        # after local training (see step 6).
        global_weights = [w.copy() for w in weights_list]

        # 2) Parse config from server
        batch_id = ins.config.get("batch_id", None)
        local_epochs = ins.config.get("local_epochs", self.local_epochs)
        proximal_mu = float(ins.config.get("proximal_mu", 0.0))

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

            # Ultralytics accumulates gradients up to a nominal batch of 64, so it
            # only calls optimizer.step() every round(64/batch) batches. On a small
            # shard an entire round can finish without a single step: training
            # "succeeds", metrics are logged, and the returned weights are bit-for-bit
            # what the server sent. Warn rather than fail — a caller may want this.
            batch = get_optimal_batch_size()
            n_train = count_shard_examples(self.batch_id, "train")
            steps = math.ceil(n_train / batch) * int(local_epochs)
            accumulate = max(round(NOMINAL_BATCH_SIZE / batch), 1)
            if steps < accumulate:
                logger.warning(
                    f"[Client] batch={batch} over {n_train} images x {local_epochs} epoch(s) "
                    f"gives {steps} batch(es), fewer than the {accumulate} needed for one "
                    f"optimizer step. This round will not change the weights. Raise "
                    f"local_epochs or lower the batch size."
                )
            results = self.yolo.train(
                data=data_yaml_path,
                epochs=local_epochs,
                imgsz=DEFAULT_IMAGE_SIZE,
                device=self.device,
                batch=batch,
                verbose=False,
                # Ultralytics defaults to 8. Inside a Ray actor on Windows that is a
                # deadlock, not a slowdown — spawned dataloader children never join.
                # With workers=0 the JPEG decode runs on the training thread, which is
                # why `cache` exists: it moves the decode out of the step loop instead
                # of trying to move it onto processes that cannot be spawned here.
                workers=0 if IS_WINDOWS else 8,
                cache=self.cache,
                # Without an explicit name every round writes runs/detect/train, train2,
                # ... at ~22 MB each, per client, forever.
                project="runs/fl",
                name=f"batch{self.batch_id}",
                exist_ok=True,
            )
            
            # 5) Process results
            if hasattr(results, "results_dict"):
                # Real shard size → FedAvg aggregation weight proportional to data.
                num_examples = count_shard_examples(self.batch_id, "train")
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

            # FedProx: pull the locally-trained weights back toward the global model
            # by factor mu, i.e. w <- w - mu * (w - w_global). mu == 0 is plain FedAvg.
            # This is a round-level approximation of the FedProx proximal term, applied
            # in weight space because Ultralytics' high-level trainer does not expose a
            # per-step loss hook. See docs/FEDPROX.md.
            if proximal_mu > 0 and len(updated_weights) == len(global_weights):
                logger.info(f"[Client] Applying FedProx proximal term (mu={proximal_mu}).")
                updated_weights = [
                    w - proximal_mu * (w - g)
                    for w, g in zip(updated_weights, global_weights)
                ]
                metrics["proximal_mu"] = proximal_mu

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
            # Real val-shard size → correct weighting of the aggregated loss.
            num_examples = count_shard_examples(self.batch_id, "val")
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
        # Read from the run config rather than from a fit instruction: FedAvg shares
        # one FitIns across every client, so anything per-client sent that way arrives
        # as the last value written (the B9 bug). The cache setting is per *run*, so
        # the run config is where it belongs.
        cache = context.run_config.get("cache", "")

        client = FlowerClient(
            model_path=model_path,
            client_state=client_state,
            local_epochs=local_epochs,
            batch_id_range=batch_id_range,
            cache=cache,
        )
        
        return client
    except Exception as e:
        logger.error(f"[Client] client_fn failed to create FlowerClient: {e}", exc_info=True)
        return None


# So you can start the client with: flwr run --app client:app
app = ClientApp(client_fn=client_fn)
