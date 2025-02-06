import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP runtime conflicts

import flwr as fl
import torch
import logging
from ultralytics import YOLO  # Ultralytics YOLO API
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from util import load_model, get_model_parameters, set_weights
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("client.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

from util import load_config

# Load configuration
config = load_config()

# Access paths
train_path = config["data"]["train"]
val_path = config["data"]["val"]
test_path = config["data"]["test"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.Client):
    def __init__(self, device, data_config, num_examples, client_id):
        super().__init__()
        self.device = device
        self.data_config = data_config
        self.num_examples = num_examples
        self.client_id = client_id
        self.model = load_model(device)  # Load YOLOv5 model

    def set_parameters(self, parameters):
        """Set model parameters from Flower server."""
        try:
            weights = parameters_to_ndarrays(parameters.parameters)
            set_weights(self.model.model, weights)  # Set weights for Ultralytics YOLO model
        except Exception as e:
            logging.error(f"Error setting parameters: {e}")
            raise

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        try:
            self.set_parameters(parameters)
            batch_size = config.get("batch_size", 16)  # Use config values
            epochs = config.get("epochs", 2)  # Use config values

            # Train the YOLOv5 model
            results = self.model.train(
                data=self.train_path,
                epochs=epochs,
                imgsz=640,
                batch=batch_size,
                device=self.device,
                save=True,  # Save checkpoints
                save_period=1,  # Save after every epoch
            )

            # Extract metrics from training results
            metrics = {
                "mp": results.results_dict["metrics/precision"],  # Precision
                "mr": results.results_dict["metrics/recall"],  # Recall
                "map50": results.results_dict["metrics/mAP_0.5"],  # mAP@0.5
                "map": results.results_dict["metrics/mAP_0.5:0.95"],  # mAP@0.5:0.95
            }

            # Return training results to the server
            return FitRes(
                status=Status(Code.OK, message="Training successful"),
                parameters=get_model_parameters(self.model.model),  # Send updated model parameters
                num_examples=self.num_examples["trainset"],  # Number of training examples
                metrics=metrics,  # Training metrics
            )
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return FitRes(
                status=Status(Code.FIT_NOT_IMPLEMENTED, message="Training failed"),
                parameters=parameters,  # Return original parameters if training fails
                num_examples=self.num_examples["trainset"],
                metrics={},  # No metrics if training fails
            )

    def evaluate(self, parameters):
        """Evaluate the model on the local validation set."""
        try:
            self.set_parameters(parameters)
            with torch.no_grad():  # Disable gradient computation
                results = self.model.val(data=self.data_config, imgsz=640, batch=16, device=self.device)

            # Extract metrics from evaluation results
            metrics = {
                "mp": results.results_dict["metrics/precision"],  # Precision
                "mr": results.results_dict["metrics/recall"],  # Recall
                "map50": results.results_dict["metrics/mAP_0.5"],  # mAP@0.5
                "map": results.results_dict["metrics/mAP_0.5:0.95"],  # mAP@0.5:0.95
            }

            # Return evaluation results to the server
            return EvaluateRes(
                status=Status(Code.OK, message="Evaluation successful"),
                loss=results.results_dict["metrics/loss"],  # Use actual loss from validation
                num_examples=self.num_examples["testset"],  # Number of test examples
                metrics=metrics,  # Evaluation metrics
            )
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return EvaluateRes(
                status=Status(Code.EVALUATE_NOT_IMPLEMENTED, message="Evaluation failed"),
                loss=0.0,  # Default loss if evaluation fails
                num_examples=self.num_examples["testset"],
                metrics={},  # No metrics if evaluation fails
            )

def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    logging.info("Shutting down client gracefully...")
    sys.exit(0)

def main():
    """Start Flower client."""
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

    data_config = "config/data.yaml"  # Path to dataset config
    num_examples = {"trainset": 80, "testset": 80}  # Example dataset sizes
    client_id = 0  # Unique ID for each client (can be passed as an argument)

    # Create and start the Flower client
    client = FlowerClient(device=DEVICE, data_config=data_config, num_examples=num_examples, client_id=client_id)

    # Connect to the Flower server on port 8081
    fl.client.start_client(server_address="0.0.0.0:8081", client=client)

if __name__ == "__main__":
    main()