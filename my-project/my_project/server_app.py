import logging
import os
import random
import warnings
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns, EvaluateIns, NDArrays, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager

# Ensure Ultralytics does not use HUB (prevents import issues)
os.environ["ULTRALYTICS_HUB"] = "0"
from ultralytics import YOLO
from my_project.task import download_model
from my_project.get_set_model import get_weights, set_weights

from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("server", "logs/server.log")

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
        logger.info(f"[Server] CustomBatchStrategy initialized with batch_id_range={batch_id_range}")
    
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
            fit_config = fit_ins.config
            try:
                # Assign a unique batch_id for this client
                batch_id = self._get_unused_batch_id(client_proxy.cid)
                
                # Insert it into the config
                fit_config["batch_id"] = batch_id
                fit_config["local_epochs"] = 1

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
            eval_config = eval_ins.config
            try:
                # Use the same batch_id assignment logic
                batch_id = self._get_unused_batch_id(client_proxy.cid)
                
                eval_config["batch_id"] = batch_id
                eval_config["local_epochs"] = 1 
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
        
        # Log results and failures
        logger.info(f"[Server] Aggregating {len(results)} fit results and {len(failures)} failures")
        
        # Call the parent's aggregation method
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if parameters is not None:
            # Calculate parameters checksum for verification
            weights = parameters_to_ndarrays(parameters)
            weights_checksum = sum(w.sum() for w in weights if w.size > 0)
            logger.info(f"[Server] Aggregated parameters with checksum: {weights_checksum}")
            
            # Add checksum to metrics for tracking
            metrics["weights_checksum"] = float(weights_checksum)
            
            # Log metrics
            logger.info(f"[Server] Round {server_round} metrics: {metrics}")
        else:
            logger.warning(f"[Server] No aggregated parameters produced in round {server_round}")
            
        return parameters, metrics


def server_fn(_):
    """
    Initialize the Flower server with a YOLO model and custom federated learning strategy.
    
    Returns:
        ServerAppComponents: Configured server components for federated learning
    """
    logger.info("[Server] Initializing YOLO model for FL...")

    # Check if model exists, otherwise download
    if not os.path.exists(MODEL_PATH):
        logger.info(f"[Server] Model not found at {MODEL_PATH}, downloading...")
        download_model()
        logger.info(f"[Server] Model successfully downloaded to {MODEL_PATH}")

    # Load YOLO's initial model
    try:
        model = YOLO(MODEL_PATH)
        initial_weights = get_weights(model)
        
        # Calculate initial checksum for tracking
        initial_checksum = sum(w.sum() for w in initial_weights if w.size > 0)
        logger.info(f"[Server] Initial model loaded with weights checksum: {initial_checksum}")
    except Exception as e:
        logger.error("[Server] Could not load YOLO model or extract weights!", exc_info=True)
        raise RuntimeError("Server cannot start without a valid YOLO model.") from e

    # Build custom strategy
    strategy = CustomBatchStrategy(
        fraction_fit=1.0,        # Use all available clients each round
        min_fit_clients=2,       # Minimum clients needed for training
        min_available_clients=2, # Minimum clients needed to start FL (consistent with min_fit_clients)
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
        batch_id_range=DEFAULT_BATCH_ID_RANGE
    )

    # Configure server
    server_config = ServerConfig(num_rounds=DEFAULT_NUM_ROUNDS)

    logger.info(f"[Server] FedAvg-based strategy configured for {DEFAULT_NUM_ROUNDS} rounds")
    logger.info("[Server] Server initialization complete and ready for clients")
    
    return ServerAppComponents(
        strategy=strategy, 
        config=server_config
    )


app = ServerApp(server_fn=server_fn)