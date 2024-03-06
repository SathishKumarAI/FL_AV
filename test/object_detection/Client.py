import torch  # Assuming torch for model
import flwr  # Assuming flwr for federated learning
import logging
from flwr.server import ClientManager
import flwr as fl
from trainer.yolov5_trainer import YOLOv5Trainer
import torch
import torchvision.transforms as transforms
import flwr as fl
import logging
from model.yolov5.utils import augmentations, loss
from load_data import load_data
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from model.init_yolo import init_yolo
from types import SimpleNamespace
def fit_config( yaml_path):
    """
    def fit_config(server_uri, client_id, yaml_path):
    Provides fit configuration for the client.
    Args:
        server_uri: The URI of the Flower server.
        client_id: The ID of the client.
        yaml_path: Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing fit configuration.
    """
    # Load YAML configuration
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with additional parameters or overrides (optional)
    return config

# print(fit_config(r"/major/test/object_detection/config/config_yolov5.yaml"))
# C:\Users\siu856522160\Downloads\major-master\test\object_detection\config\config_yolov5.yaml
args = fit_config(r"C:\Users\siu856522160\Downloads\major\test\object_detection\config\config_yolov5.yaml")
args = SimpleNamespace(**args)
# Print key-value pairs
for key, value in vars(args).items():
    print(f"{key}: {value}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model, dataset, dataset_train, dataset_val, trainer, args = init_yolo(args=args, device=DEVICE)
model, dataset, dataset_train, trainer, args = init_yolo(args=args, device=DEVICE)

# print(f"model: {model}")
print(f"dataset: {dataset}")
print(f"dataset: {dataset_train}")
print(type(dataset_train))
# print(f"dataset: {dataset_val}")
print(f"trainer: {trainer}")
print(f"args: {args}")
from trainer.yolov5_trainer import YOLOv5Trainer
# train(self, train_data, device, args)
def main():

    # Start Flower server
    fl.client.start_client(
        server_address="0.0.0.0:8080",
        # config=fl.server.ServerConfig(num_rounds=3),
        client= YOLOv5Trainer(model, args=args)
        # client= YOLOv5Trainer(model, args=args).train(train_data=dataset_train, device=DEVICE, args=args)# best code
        # client_fn= YOLOv5Trainer(model, args=args).train(train_data=dataset_train, device=DEVICE, args=args)
        )

if __name__ == "__main__":
    main()
