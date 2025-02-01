import logging
import os
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from ultralytics import YOLO
from my_project.task import get_weights

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Centralized logging setup
from utils.logging_setup import configure_logging

logger = configure_logging("server", "logs/server.log")

def server_fn(context: Context):
    try:
        config = {
            # 'num_server_rounds': context.run_config["num_server_rounds"],
            # 'fraction_fit': context.run_config["fraction_fit"],
            'num_rounds':5,
            'fraction_fit':0.75,
            'min_clients': 4
        }

        # model_path = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\models\yolov5su.pt"
        model = YOLO("yolov5su.pt")
        if model is None:
            # raise ValueError("Model failed to load")
            logger.error(f"Model failed to load")
        logger.info(f"Loaded YOLOv5 model from model_path")

        strategy = FedAvg(
            fraction_fit=config['fraction_fit'],
            min_fit_clients=2,  # Ensure training proceeds with at least 2 clients
            min_available_clients=max(2, int(config['min_clients'])),  # Ensure enough clients
            # min_available_clients=2,
            initial_parameters=ndarrays_to_parameters(get_weights(model)),
        )
        logger.info(f"Loaded  strategy  from model_path")
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=config['num_rounds'])
        )
    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}", exc_info=True)
        raise

app = ServerApp(server_fn=server_fn)
