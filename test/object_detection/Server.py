import flwr as fl
import logging
import yaml
import torch 
from load_data import load_data
from model.init_yolo import init_yolo
from model.yolov5.models.yolo import Model as YOLOv5


# Configure and run the server
def main():
    fl.server.start_server(
        server_address ="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        
        )

if __name__ == "__main__":
    main()
