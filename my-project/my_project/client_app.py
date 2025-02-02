import logging
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from ultralytics import YOLO
from my_project.task import get_weights, set_weights, train, test, load_data

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Centralized logging setup
from utils.logging_setup import configure_logging

class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs, batch_id):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = configure_logging(f"client_{batch_id}", f"logs/client_{batch_id}.log")
        self.logger.info(f"Client {batch_id} initialized on {self.device}")

    def fit(self, parameters, config):
        try:
            if not set_weights(self.model, parameters):
                raise ValueError("Failed to set model weights")

            self.model.to(self.device)
            train_loss = train(self.model, self.trainloader, self.local_epochs, self.device)

            return get_weights(self.model), len(self.trainloader.dataset), {"train_loss": train_loss}
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return parameters, 0, {"train_loss": float("inf")}

    def evaluate(self, parameters, config):
        try:
            if not set_weights(self.model, parameters):
                raise ValueError("Failed to set evaluation weights")

            loss, accuracy = test(self.model, self.valloader, self.device)
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return float("inf"), 0, {}

def client_fn(context: Context):
    try:
        # batch_id = context.node_config["partition-id"]
        batch_id = 3
        logger = configure_logging(f"client_{batch_id}", f"logs/client_{batch_id}.log")
        logger.info(f"Client {batch_id} starting...")

        if not load_data(batch_id, "train") or not load_data(batch_id, "val"):
            logger.warning(f"Skipping client - Invalid data for batch {batch_id}")
            return None

        model = YOLO("yolov5su.pt")
        trainloader = load_data(batch_id, "train")
        valloader = load_data(batch_id, "val")

        return FlowerClient(model, trainloader, valloader, context.run_config.get("local-epochs", 3), batch_id).to_client()
    except Exception as e:
        logger.error(f"Client initialization failed: {str(e)}")
        return None

app = ClientApp(client_fn=client_fn)
