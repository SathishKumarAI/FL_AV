import logging
import os
import random
import warnings
import platform
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context, Parameters, FitIns, EvaluateIns, NDArrays, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager

# Ensure Ultralytics does not use HUB (prevents import issues)
os.environ["ULTRALYTICS_HUB"] = "0"
from ultralytics import YOLO
from my_project.task import download_model, should_checkpoint, IS_WINDOWS, OS_NAME
from my_project.get_set_model import NUM_CLASSES_MODEL_YAML, get_weights, set_weights

from utils.logging_setup import configure_logging
from utils.metrics_logger import MetricsLogger, aggregate_client_metrics

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("server", "logs/server.log")

# Log OS detection
logger.info(f"[Server] Detected operating system: {OS_NAME}, IS_WINDOWS={IS_WINDOWS}")

# Constants
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
DEFAULT_BATCH_ID_RANGE = (1, 10)
DEFAULT_NUM_ROUNDS = 3

class CustomBatchStrategy(FedAvg):
    """
    A FedAvg-based strategy that dynamically assigns each client a unique 'batch_id'
    in configure_fit() and configure_evaluate(), ensuring each client uses a different data.yaml.
    
    Attributes:
        batch_id_range (tuple): Min and max range for batch IDs (inclusive)
        used_batch_ids (set): Tracks batch IDs used in the current round
        client_to_batch_id (dict): Maps client IDs to their assigned batch IDs
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Any] = None,
        on_fit_config_fn: Optional[Any] = None,
        on_evaluate_config_fn: Optional[Any] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Any] = None,
        evaluate_metrics_aggregation_fn: Optional[Any] = None,
        batch_id_range: tuple = DEFAULT_BATCH_ID_RANGE,
        proximal_mu: float = 0.0,
        num_rounds: int = DEFAULT_NUM_ROUNDS,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.batch_id_range = batch_id_range
        self.used_batch_ids = set()
        self.client_to_batch_id: Dict[str, int] = {}
        self.client_os_info: Dict[str, str] = {}  # Track client OS information
        self.num_rounds = num_rounds
        self.metrics_logger = MetricsLogger()  # writes logs/metrics.csv

        # Global-model checkpointing: persist the aggregated model to disk so the
        # federated result survives process exit. Lazily instantiate one YOLO to
        # hold/save weights (created on first save to avoid a redundant load).
        self.checkpoint_dir = checkpoint_dir
        self.save_every = max(int(save_every), 1)
        self._save_model = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # proximal_mu > 0 turns on FedProx-style proximal regularization on clients.
        self.proximal_mu = float(proximal_mu)
        strategy_name = "FedProx" if self.proximal_mu > 0 else "FedAvg"
        logger.info(
            f"[Server] CustomBatchStrategy initialized with batch_id_range={batch_id_range}, "
            f"strategy={strategy_name} (proximal_mu={self.proximal_mu})"
        )
    
    def _get_unused_batch_id(self, client_id: str) -> int:
        """
        Get a batch_id that hasn't been used yet in the current round.
        
        Args:
            client_id: The client's identifier
            
        Returns:
            int: A unique batch ID for this client
        """
        # Check if client already has a batch_id assigned
        if client_id in self.client_to_batch_id:
            logger.debug(f"[Server] Client {client_id} already has batch_id={self.client_to_batch_id[client_id]}")
            return self.client_to_batch_id[client_id]
            
        min_id, max_id = self.batch_id_range
        available_ids = set(range(min_id, max_id + 1)) - self.used_batch_ids
        
        if not available_ids:
            # If all batches have been used, log warning and reset tracking
            logger.warning(f"[Server] All batch_ids in range {min_id}-{max_id} have been used. Resetting usage tracking.")
            self.used_batch_ids = set()  # Reset used batches
            available_ids = set(range(min_id, max_id + 1))
            
        batch_id = random.choice(list(available_ids))
        self.used_batch_ids.add(batch_id)
        self.client_to_batch_id[client_id] = batch_id
        logger.debug(f"[Server] Assigned new batch_id={batch_id} to client {client_id}")
        return batch_id
    
    def _clear_round_state(self) -> None:
        """Clear state that should be reset between rounds."""
        logger.debug(f"[Server] Clearing round state, resetting {len(self.used_batch_ids)} used batch IDs")
        self.used_batch_ids = set()

    def _save_global_model(self, weights, server_round: int) -> None:
        """
        Save the aggregated global weights as a self-contained YOLO checkpoint.

        Loads the ndarray weights into a held YOLO model (via set_weights, which
        now includes BatchNorm buffers) and writes both a per-round checkpoint and
        a stable ``global_last.pt`` pointer. Failures are logged, never fatal —
        a checkpointing error must not abort the federation.
        """
        try:
            if self._save_model is None:
                self._save_model = YOLO(NUM_CLASSES_MODEL_YAML).load(MODEL_PATH)
            if not set_weights(self._save_model.model, weights):
                logger.error(f"[Server] Round {server_round}: set_weights failed; skipping checkpoint.")
                return
            round_path = os.path.join(self.checkpoint_dir, f"global_round_{server_round}.pt")
            last_path = os.path.join(self.checkpoint_dir, "global_last.pt")
            self._save_model.save(round_path)
            self._save_model.save(last_path)
            logger.info(f"[Server] Saved global checkpoint: {round_path} (and {last_path})")
        except Exception as e:
            logger.error(f"[Server] Failed to save global checkpoint at round {server_round}: {e}", exc_info=True)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure training by assigning unique batch_id to each client.
        
        Args:
            server_round: Current round number
            parameters: Model parameters to distribute
            client_manager: Manages available clients
            
        Returns:
            List of tuples containing client proxies and their fit instructions
        """
        logger.info(f"[Server] configure_fit: Round={server_round}. Assigning batch IDs to clients...")
        self._clear_round_state()  # Reset state for new round

        # Log parameter information for debugging
        weights = parameters_to_ndarrays(parameters)
        weights_checksum = sum(w.sum() for w in weights if w.size > 0)
        logger.info(f"[Server] Sending parameters with checksum: {weights_checksum}")

        # Delegate to FedAvg for the initial instructions
        instructions = super().configure_fit(server_round, parameters, client_manager)
        logger.info(f"[Server] Configured {len(instructions)} clients for training in round {server_round}")

        updated_instructions = []
        for (client_proxy, fit_ins) in instructions:
            # COPY, do not mutate. FedAvg.configure_fit builds ONE FitIns and hands
            # the same object to every client, so writing batch_id into fit_ins.config
            # in this loop overwrote it for all of them — last client won and the whole
            # federation trained a single shard while the log claimed otherwise.
            fit_config = dict(fit_ins.config)
            try:
                # Assign a unique batch_id for this client
                batch_id = self._get_unused_batch_id(client_proxy.cid)
                
                # Insert batch_id into the config. local_epochs is supplied by
                # on_fit_config_fn (driven by run_config), so we do not override it here.
                fit_config["batch_id"] = batch_id
                # Tell the client how strongly to pull local weights back toward the
                # global model (0.0 => plain FedAvg, no proximal term).
                fit_config["proximal_mu"] = self.proximal_mu

                logger.info(
                    f"[Server] Assigning batch_id={batch_id} "
                    f"to client {client_proxy.cid} in round={server_round}"
                )

                # Recreate the FitIns with the updated config
                new_fit_ins = FitIns(parameters=fit_ins.parameters, config=fit_config)
                updated_instructions.append((client_proxy, new_fit_ins))

            except Exception as e:
                logger.error(
                    f"[Server] Failed to assign batch_id for client {client_proxy.cid}: {e}",
                    exc_info=True
                )
                updated_instructions.append((client_proxy, fit_ins))  # fallback

        return updated_instructions
    
    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure evaluation by assigning unique batch_id to each client.
        
        Args:
            server_round: Current round number
            parameters: Model parameters to distribute
            client_manager: Manages available clients
            
        Returns:
            List of tuples containing client proxies and their evaluation instructions
        """
        logger.info(f"[Server] configure_evaluate: Round={server_round}. Assigning batch IDs...")

        # Log parameter information for debugging
        weights = parameters_to_ndarrays(parameters)
        weights_checksum = sum(w.sum() for w in weights if w.size > 0)
        logger.info(f"[Server] Sending evaluation parameters with checksum: {weights_checksum}")

        # Use default Flower behavior
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        logger.info(f"[Server] Configured {len(instructions)} clients for evaluation in round {server_round}")

        updated_instructions = []
        for (client_proxy, eval_ins) in instructions:
            eval_config = dict(eval_ins.config)  # shared EvaluateIns — copy, see configure_fit
            try:
                # Use the same batch_id assignment logic
                batch_id = self._get_unused_batch_id(client_proxy.cid)
                
                eval_config["batch_id"] = batch_id
                logger.info(f"[Server] Assigned batch_id={batch_id} to client {client_proxy.cid} for evaluation.")

                # Create new EvaluateIns with updated config
                new_eval_ins = EvaluateIns(parameters=eval_ins.parameters, config=eval_config)
                updated_instructions.append((client_proxy, new_eval_ins))

            except Exception as e:
                logger.error(f"[Server] Failed to assign batch_id for evaluation: {e}", exc_info=True)
                updated_instructions.append((client_proxy, eval_ins))  # Fallback

        return updated_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitIns]],
        failures: List[Union[Tuple[ClientProxy, FitIns], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model parameters and training metrics from client training results.
        
        Args:
            server_round: Current round number
            results: Training results from each client
            failures: List of client failures
            
        Returns:
            Tuple of (aggregated parameters, metrics dict) 
        """
        if not results:
            logger.warning(f"[Server] No results to aggregate in round {server_round}")
            return None, {}
        
        # Track OS information from clients
        for client_proxy, fit_res in results:
            if "os" in fit_res.metrics:
                client_os = fit_res.metrics["os"]
                self.client_os_info[client_proxy.cid] = client_os
                logger.info(f"[Server] Client {client_proxy.cid} is running on {client_os}")
        
        # Log results and failures
        logger.info(f"[Server] Aggregating {len(results)} fit results and {len(failures)} failures")

        # Persist a weighted-average of client training metrics for this round.
        fit_metrics = aggregate_client_metrics(results)
        self.metrics_logger.log_round(server_round, "fit", fit_metrics, num_clients=len(results))

        # Call the parent's aggregation method
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if parameters is not None:
            # Calculate parameters checksum for verification
            weights = parameters_to_ndarrays(parameters)
            weights_checksum = sum(w.sum() for w in weights if w.size > 0)
            logger.info(f"[Server] Aggregated parameters with checksum: {weights_checksum}")

            # Persist the aggregated global model on the configured cadence and on
            # the final round, so the federated result is recoverable after exit.
            if should_checkpoint(server_round, self.save_every, self.num_rounds):
                self._save_global_model(weights, server_round)

            # Add checksum and OS counts to metrics for tracking
            metrics["weights_checksum"] = float(weights_checksum)
            metrics["server_os"] = OS_NAME
            
            # Count clients by OS
            os_counts = {}
            for os_name in self.client_os_info.values():
                os_counts[f"client_count_{os_name}"] = os_counts.get(f"client_count_{os_name}", 0) + 1
            
            # Add OS counts to metrics
            metrics.update({k: float(v) for k, v in os_counts.items()})
            
            # Log metrics
            logger.info(f"[Server] Round {server_round} metrics: {metrics}")
        else:
            logger.warning(f"[Server] No aggregated parameters produced in round {server_round}")
            
        return parameters, metrics
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitIns]],
        failures: List[Union[Tuple[ClientProxy, FitIns], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Evaluation results from each client
            failures: List of client failures
            
        Returns:
            Tuple of (loss, metrics dict)
        """
        if not results:
            return None, {}
            
        # Track OS information from clients
        for client_proxy, eval_res in results:
            if "os" in eval_res.metrics:
                client_os = eval_res.metrics["os"]
                self.client_os_info[client_proxy.cid] = client_os
                logger.info(f"[Server] Client {client_proxy.cid} evaluated on {client_os}")
        
        # Persist a weighted-average of client evaluation metrics for this round.
        eval_metrics = aggregate_client_metrics(results)

        # Call the parent's aggregation method
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.metrics_logger.log_round(
            server_round, "evaluate", eval_metrics, num_clients=len(results), loss=loss
        )
        # Emit the run summary once the final round has been evaluated.
        if server_round >= self.num_rounds:
            self.metrics_logger.summary()

        # Add OS information to metrics
        metrics["server_os"] = OS_NAME
        
        # Count clients by OS
        os_counts = {}
        for os_name in self.client_os_info.values():
            os_counts[f"client_count_{os_name}"] = os_counts.get(f"client_count_{os_name}", 0) + 1
        
        # Add OS counts to metrics
        metrics.update({k: float(v) for k, v in os_counts.items()})
        
        return loss, metrics


def server_fn(context: Context):
    """
    Initialize the Flower server with a YOLO model and custom federated learning strategy.

    Hyperparameters are read from ``context.run_config`` (defined in pyproject.toml under
    ``[tool.flwr.app.config]``) instead of being hardcoded, so they can be overridden with
    ``flwr run --run-config "num_server_rounds=5 local_epochs=2"``.

    Returns:
        ServerAppComponents: Configured server components for federated learning
    """
    logger.info("[Server] Initializing YOLO model for FL...")

    # Read run configuration (with safe fallbacks).
    run_config = getattr(context, "run_config", {}) or {}
    num_rounds = int(run_config.get("num_server_rounds", DEFAULT_NUM_ROUNDS))
    fraction_fit = float(run_config.get("fraction_fit", 1.0))
    local_epochs = int(run_config.get("local_epochs", 1))
    min_clients = int(run_config.get("min_clients", 2))
    # Strategy selection from run_config: strategy = "fedavg" | "fedprox".
    strategy_name = str(run_config.get("strategy", "fedavg")).lower()
    proximal_mu = float(run_config.get("proximal_mu", 0.0))
    if strategy_name == "fedprox" and proximal_mu <= 0:
        proximal_mu = 0.1  # sensible default when FedProx is requested without a mu
    if strategy_name != "fedprox":
        proximal_mu = 0.0  # force plain FedAvg
    checkpoint_dir = str(run_config.get("checkpoint_dir", "checkpoints"))
    save_every = int(run_config.get("save_every", 1))
    logger.info(
        f"[Server] run_config -> num_rounds={num_rounds}, fraction_fit={fraction_fit}, "
        f"local_epochs={local_epochs}, min_clients={min_clients}, "
        f"strategy={strategy_name}, proximal_mu={proximal_mu}"
    )

    # Check if model exists, otherwise download
    if not os.path.exists(MODEL_PATH):
        logger.info(f"[Server] Model not found at {MODEL_PATH}, downloading...")
        download_model()
        logger.info(f"[Server] Model successfully downloaded to {MODEL_PATH}")

    # Load YOLO's initial model
    try:
        # Same 13-class arch the clients build (client_app.py). Building from the
        # .pt alone gives an 80-class COCO head, which stops matching the moment a
        # client's train() rebuilds its head from data.yaml (nc=13). Pass the inner
        # DetectionModel, not the YOLO wrapper, so both sides key the state_dict
        # identically.
        model = YOLO(NUM_CLASSES_MODEL_YAML).load(MODEL_PATH)
        initial_weights = get_weights(model.model)
        
        # Calculate initial checksum for tracking
        initial_checksum = sum(w.sum() for w in initial_weights if w.size > 0)
        logger.info(f"[Server] Initial model loaded with weights checksum: {initial_checksum}")
    except Exception as e:
        logger.error("[Server] Could not load YOLO model or extract weights!", exc_info=True)
        raise RuntimeError("Server cannot start without a valid YOLO model.") from e

    # Push local_epochs to every client each round via config callbacks.
    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {"local_epochs": local_epochs, "server_round": server_round}

    # Build custom strategy
    strategy = CustomBatchStrategy(
        fraction_fit=fraction_fit,            # From run_config
        min_fit_clients=min_clients,          # From run_config
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,    # Minimum clients needed to start FL
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
        batch_id_range=DEFAULT_BATCH_ID_RANGE,
        proximal_mu=proximal_mu,
        num_rounds=num_rounds,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
    )

    # Configure server
    server_config = ServerConfig(num_rounds=num_rounds)

    logger.info(f"[Server] FedAvg-based strategy configured for {num_rounds} rounds")
    logger.info(f"[Server] Server running on {OS_NAME} is ready for clients")
    
    return ServerAppComponents(
        strategy=strategy, 
        config=server_config
    )


app = ServerApp(server_fn=server_fn)