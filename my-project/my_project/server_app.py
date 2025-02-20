import logging
import os
import warnings
import torch

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns, EvaluateIns, ndarrays_to_parameters, Context
from typing import List, Tuple
# Ensure Ultralytics does not use HUB (prevents import issues)
os.environ["ULTRALYTICS_HUB"] = "0"
from ultralytics import YOLO
from my_project.task import  download_model  # If needed, though we only do get_weights here
from my_project.get_set_model import get_weights, load_yolo_model, set_weights

from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("server", "logs/server.log")

import os
import urllib.request
import logging
# Correct Model Path and URL
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
# Setup logger
logger = logging.getLogger("server")

class CustomBatchStrategy(FedAvg):
    """
    A FedAvg-based strategy that dynamically assigns each client a unique 'batch_id'
    in configure_fit(), ensuring each client uses a different data.yaml.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.next_batch_id = 1  # Start assigning from batch_id=1

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Called each round by Flower to decide which clients train (fit) and how.
        We'll:
        1) Use FedAvg to get the default instructions,
        2) For each selected client, inject a unique batch_id into fit_ins.config.
        """
        logger.info(f"[Server] configure_fit: Round={server_round}. Assigning batch IDs to clients...")

        # Delegate to FedAvg for the initial instructions
        instructions = super().configure_fit(server_round, parameters, client_manager)

        updated_instructions = []
        for (client_proxy, fit_ins) in instructions:
            fit_config = fit_ins.config  # This is a Dict[str, Scalar]
            try:
                # 1) Assign a unique batch_id
                batch_id_assigned = self.next_batch_id
                self.next_batch_id += 1

                # 2) Insert it into the config
                fit_config["batch_id"] = batch_id_assigned

                # Example: also specify a local_epochs if desired
                fit_config["local_epochs"] = 1  # or something dynamic

                logger.info(
                    f"[Server] Assigning batch_id={batch_id_assigned} "
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
            client_manager: fl.server.client_manager.ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure evaluation by assigning batch_id to each client.
        """
        logger.info(f"[Server] configure_evaluate: Round={server_round}. Assigning batch IDs...")

        # Use default Flower behavior
        instructions = super().configure_evaluate(server_round, parameters, client_manager)

        updated_instructions = []
        for (client_proxy, eval_ins) in instructions:
            eval_config = eval_ins.config  # This is a Dict[str, Scalar]
            try:
                # Assign batch_id (just increment from training)
                batch_id_assigned = self.next_batch_id
                self.next_batch_id += 1

                eval_config["batch_id"] = batch_id_assigned  # Inject batch_id

                logger.info(f"[Server] Assigned batch_id={batch_id_assigned} to client {client_proxy.cid} for evaluation.")

                # Create new EvaluateIns with updated config
                new_eval_ins = EvaluateIns(parameters=eval_ins.parameters, config=eval_config)
                updated_instructions.append((client_proxy, new_eval_ins))

            except Exception as e:
                logger.error(f"[Server] Failed to assign batch_id for evaluation: {e}", exc_info=True)
                updated_instructions.append((client_proxy, eval_ins))  # Fallback

        return updated_instructions


def server_fn(context: Context):
    """
    Flower calls this function at startup to get the server strategy/config.
    """
    logger.info("[Server] Initializing YOLO model for FL...")

    # Check if model exists, otherwise download
    if not os.path.exists(MODEL_PATH):
        download_model()
    
    num_rounds = context.run_config["num_server_rounds"]
    fraction_fit = context.run_config["fraction_fit"]

    # # Load YOLO's initial model
    # try:
    #     model = YOLO(MODEL_PATH)
    #     initial_weights = get_weights(model)
    #     print(initial_weights)
    # except Exception as e:
    #     logger.error("[Server] Could not load YOLO model or extract weights!", exc_info=True)
    #     raise RuntimeError("Server cannot start without a valid YOLO model.") from e
    ndarrays = get_weights(load_yolo_model("models/yolov8s.pt"))
    # parameters = ndarrays_to_parameters(ndarrays)
    # Build custom strategy
    strategy = CustomBatchStrategy(
        fraction_fit=1.0,        # For demonstration, we use all clients each round
        min_fit_clients=1,       # Minimum 1 client needed to proceed
        min_available_clients=2, # Start FL with 1 client
        initial_parameters=fl.common.ndarrays_to_parameters(ndarrays)
    )
    config = ServerConfig(num_rounds=num_rounds)
    # We'll do 3 rounds for demonstration
    # server_config = ServerConfig(num_rounds=3)

    logger.info("[Server] FedAvg-based strategy with dynamic batch assignment is ready.")
    return ServerAppComponents(
        strategy=strategy, 
        config=config,
                            )


app = ServerApp(server_fn=server_fn)

