import logging
import os
import random
import warnings
from typing import List, Tuple, Dict

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns, EvaluateIns
from flwr.server.client_manager import ClientManager

# Ensure Ultralytics does not use HUB (prevents import issues)
os.environ["ULTRALYTICS_HUB"] = "0"
from ultralytics import YOLO
from my_project.task import download_model
from my_project.get_set_model import get_weights, set_weights

from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("server", "logs/server.log")

# Correct Model Path and URL
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"

class CustomBatchStrategy(FedAvg):
    """
    A FedAvg-based strategy that dynamically assigns each client a unique 'batch_id'
    in configure_fit() and configure_evaluate(), ensuring each client uses a different data.yaml.
    
    Attributes:
        batch_id_range (tuple): Min and max range for batch IDs (inclusive)
        used_batch_ids (set): Tracks batch IDs used in the current round
        client_to_batch_id (dict): Maps client IDs to their assigned batch IDs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_id_range = (1, 10)
        self.used_batch_ids = set()
        self.client_to_batch_id: Dict[str, int] = {}
    
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
        return batch_id
    
    def _clear_round_state(self) -> None:
        """Clear state that should be reset between rounds."""
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

        # Delegate to FedAvg for the initial instructions
        instructions = super().configure_fit(server_round, parameters, client_manager)

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

        # Use default Flower behavior
        instructions = super().configure_evaluate(server_round, parameters, client_manager)

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


def server_fn(_):
    """
    Initialize the Flower server with a YOLO model and custom federated learning strategy.
    
    Returns:
        ServerAppComponents: Configured server components for federated learning
    """
    logger.info("[Server] Initializing YOLO model for FL...")

    # Check if model exists, otherwise download
    if not os.path.exists(MODEL_PATH):
        download_model()

    # Load YOLO's initial model
    try:
        model = YOLO(MODEL_PATH)
        initial_weights = get_weights(model)
    except Exception as e:
        logger.error("[Server] Could not load YOLO model or extract weights!", exc_info=True)
        raise RuntimeError("Server cannot start without a valid YOLO model.") from e

    # Build custom strategy
    strategy = CustomBatchStrategy(
        fraction_fit=1.0,        # Use all available clients each round
        min_fit_clients=2,       # Minimum clients needed for training
        min_available_clients=2, # Minimum clients needed to start FL (consistent with min_fit_clients)
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights)
    )

    # We'll do 3 rounds for demonstration
    server_config = ServerConfig(num_rounds=3)

    logger.info("[Server] FedAvg-based strategy with dynamic batch assignment is ready.")
    return ServerAppComponents(
        strategy=strategy, 
        config=server_config
    )


app = ServerApp(server_fn=server_fn)