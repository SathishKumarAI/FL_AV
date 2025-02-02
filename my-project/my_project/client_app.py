import logging
import torch
from flwr.client import ClientApp, Client
from flwr.common import Context
from ultralytics import YOLO
from my_project.task import get_weights, set_weights, train, test, load_data

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Centralized logging setup
from utils.logging_setup import configure_logging

logger = configure_logging("client", "logs/client.log")

class FlowerClient(Client):
    def __init__(self, model, local_epochs, batch_id):
        self.model = model
        # self.trainloader = trainloader
        # self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = configure_logging(f"client_{batch_id}", f"logs/client_{batch_id}.log")
        
        self.logger.info(f"Client {batch_id} initialized on {self.device}")
        self.logger.debug("Model,  assigned.")

    def fit(self, parameters, config):
        try:
            self.logger.info("Starting training...")
            self.logger.debug("Setting model weights...")
            if not set_weights(self.model, parameters):
                raise ValueError("Failed to set model weights")
            
            self.logger.info("Model weights set successfully. Moving model to device...")
            self.model.to(self.device)
            self.logger.debug("Model moved to device.")
            
            # self.logger.info(f"Training for {self.local_epochs} epochs on {len(self.trainloader.dataset)} samples")
            train_loss = train(self.model, self.device)
            self.logger.debug(f"Training loss recorded: {train_loss}")
            
            self.logger.info(f"Training completed. Final loss: {train_loss}")
            return get_weights(self.model), len(self.trainloader.dataset), {"train_loss": train_loss}
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            return parameters, 0, {"train_loss": float("inf")}

    def evaluate(self, parameters, config):
        try:
            self.logger.info("Starting evaluation...")
            self.logger.debug("Setting evaluation weights...")
            if not set_weights(self.model, parameters):
                raise ValueError("Failed to set evaluation weights")
            
            self.logger.info("Model weights set successfully for evaluation.")
            loss, accuracy = test(self.model, self.valloader, self.device)
            self.logger.debug(f"Evaluation metrics - Loss: {loss}, Accuracy: {accuracy}")
            
            self.logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            return float("inf"), 0, {}

def client_fn(context: Context):
    try:
        batch_id = 3  # Normally, you would get this dynamically
        logger = configure_logging(f"client_{batch_id}", f"logs/client_{batch_id}.log")
        
        logger.info(f"Client {batch_id} starting...")
        logger.debug("Loading training and validation data...")
        
        if not load_data(batch_id, "train"):
            logger.warning(f"Skipping client - Invalid training data for batch {batch_id}")
            return None
        if not load_data(batch_id, "val"):
            logger.warning(f"Skipping client - Invalid validation data for batch {batch_id}")
            return None
        
        logger.info("Data loaded successfully. Initializing model...")
        model = YOLO("yolov5su.pt")
        
        logger.debug("Loading data loaders...")
        trainloader = load_data(batch_id, "train")
        valloader = load_data(batch_id, "val")
        
        logger.info("Model initialized. Creating FlowerClient...")
        client = FlowerClient(model, trainloader, valloader, context.run_config.get("local-epochs", 3), batch_id)
        
        logger.info("FlowerClient successfully created and returning to client app.")
        return client
    except Exception as e:
        logger.error(f"Client initialization failed: {str(e)}", exc_info=True)
        return None

app = ClientApp(client_fn=client_fn)
