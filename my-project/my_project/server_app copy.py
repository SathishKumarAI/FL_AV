import logging
import os
import random
import warnings
# import numpy as np
import torch
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns, EvaluateIns,Context
from typing import List, Tuple
# Ensure Ultralytics does not use HUB (prevents import issues)
os.environ["ULTRALYTICS_HUB"] = "0"
from ultralytics import YOLO
from my_project.task import  download_model  # If needed, though we only do get_weights here
from my_project.get_set_model import get_weights, load_yolo_model, set_weights

from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("server", "logs/server.log")

# Correct Model Path and URL
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"


class CustomBatchStrategy(FedAvg):
    """
    A FedAvg-based strategy that dynamically assigns each client a unique 'batch_id'
    in configure_fit(), ensuring each client uses a different data.yaml.
    """

    def __init__(self,batch_id_range=(1,10), **kwargs):
        super().__init__(**kwargs)
        # self.next_batch_id = 1  # Start assigning from batch_id=1
        self.batch_id_range = batch_id_range
        # self.batch_id_range = (1, 10)
        # print(self.batch_id_range)
        self.client_to_batch_id = {}
        
        
        # print(self.client_to_batch_id)
        
        
    # def _get_batch_id_for_client(self, client_proxy: ClientProxy) -> int:
    #     """
    #     Assign or retrieve a consistent batch ID for a client within the valid range.
    #     """
    #     client_id = client_proxy.cid
        
    #     # If client already has an assigned batch ID, return it
    #     if client_id in self.client_to_batch_id:
    #         return self.client_to_batch_id[client_id]
        
    #     # Assign a new batch ID within range
    #     min_id, max_id = self.batch_id_range
        
    #     # Simple deterministic assignment based on client ID hash
    #     # This ensures the same client always gets the same batch ID
    #     try:
    #         # Try to convert client ID to integer if possible
    #         numeric_id = int(client_id)
    #         batch_id = (numeric_id % (max_id - min_id + 1)) + min_id
    #     except ValueError:
    #         # If client ID is not numeric, use hash
    #         hash_val = hash(client_id)
    #         batch_id = (abs(hash_val) % (max_id - min_id + 1)) + min_id
        
    #     self.client_to_batch_id[client_id] = batch_id
    #     logger.info(f"[Server] Assigned fixed batch_id={batch_id} to client {client_id}")
    #     return batch_id
    
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
                # # 1) Assign a unique batch_id
                # batch_id_assigned = self.next_batch_id
                # self.next_batch_id += 1

                # # 2) Insert it into the config
                # fit_config["batch_id"] = batch_id_assigned

                # # Example: also specify a local_epochs if desired
                # fit_config["local_epochs"] = 1  # or something dynamic

                # logger.info(
                #     f"[Server] Assigning batch_id={batch_id_assigned} "
                #     f"to client {client_proxy.cid} in round={server_round}"
                # )

                # # Recreate the FitIns with the updated config
                # new_fit_ins = FitIns(parameters=fit_ins.parameters, config=fit_config)
                # updated_instructions.append((client_proxy, new_fit_ins))
                # batch_id = self._get_batch_id_for_client(client_proxy)
                batch_id = random.randint(1,10)
                
                # Insert it into the config
                fit_config["batch_id"] = batch_id
                fit_config["local_epochs"] = 1  # or something dynamic
                fit_config["batch_id_range"] = self.batch_id_range  # Pass range to client
                logger.info(
                    f"[Server] Using batch_id={batch_id} "
                    f"for client {client_proxy.cid} in round={server_round}"
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
                # # Assign batch_id (just increment from training)
                # batch_id_assigned = self.next_batch_id
                # self.next_batch_id += 1

                # eval_config["batch_id"] = batch_id_assigned  # Inject batch_id

                # logger.info(f"[Server] Assigned batch_id={batch_id_assigned} to client {client_proxy.cid} for evaluation.")

                # # Create new EvaluateIns with updated config
                # new_eval_ins = EvaluateIns(parameters=eval_ins.parameters, config=eval_config)
                # updated_instructions.append((client_proxy, new_eval_ins))
                # Use the same batch ID as in training
                # batch_id = self._get_batch_id_for_client(client_proxy)
                batch_id = random.randint(1,10)
                # logger.info(f"[Server] Using consistent batch_id={batch_id} for client {client_proxy.cid} in evaluation.")
                eval_config["batch_id"] = batch_id
                eval_config["batch_id_range"] = self.batch_id_range  # Pass range to client

                logger.info(f"[Server] Using consistent batch_id={batch_id} for client {client_proxy.cid} in evaluation.")

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
    batch_id_range = context.run_config.get("batch_id_range", (1, 10))
    num_rounds = context.run_config["num_server_rounds"]
    # fraction_fit = context.run_config["fraction_fit"]
    
    # Get batch ID range from config, default to (1, 10)
    # batch_id_range = context.run_config.get("batch_id_range", (1, 10))

    # ndarrays = get_weights(load_yolo_model("models/yolov8s.pt"))
    # Load YOLO's initial model
    try:
        model = YOLO(MODEL_PATH)
        print("in model block")
        initial_weights = get_weights(model)
        # print(initial_weights)
        if initial_weights is None:
            print("wrong")
    except Exception as e:
        logger.error("[Server] Could not load YOLO model or extract weights!", exc_info=True)
        raise RuntimeError("Server cannot start without a valid YOLO model.") from e

    # Build custom strategy
    strategy = CustomBatchStrategy(
        batch_id_range=batch_id_range,  
        fraction_fit=1.0,        # For demonstration, we use all clients each round
        min_fit_clients=2,       # Minimum 1 client needed to proceed
        min_available_clients=3, # Start FL with 1 client
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights)
    )

    # We'll do 3 rounds for demonstration
    # server_config = ServerConfig(num_rounds=3)
    config = ServerConfig(num_rounds=num_rounds)
    logger.info("[Server] FedAvg-based strategy with dynamic batch assignment is ready.")
    return ServerAppComponents(
        strategy=strategy, 
        config=config,
        )


app = ServerApp(server_fn=server_fn)

