import torch
import os 
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict

import flwr as fl
from flwr.common import ndarrays_to_parameters

from utils.general import check_img_size, check_dataset
import warnings

from model.init_yolo import init_yolo
import util
import train
import val

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    with open("path.yaml", "r") as file:
        paths = yaml.safe_load(file)["paths"]

    model_weights_path = paths["model_weights"]
    yolov5_weights = paths["yolov5_weights"]
    data_config_path = paths["data_config"]
    ip_address = paths["ip_address"]
    
    args = util.fit_config_file(data_config_path)
    args = SimpleNamespace(**args)
    
    opt = train.parse_opt(True)
    model, dataset_train, dataset_val, args = init_yolo(args=args, device="cuda:0")
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=util.fit_config,
        initial_parameters=ndarrays_to_parameters(model_parameters),
    )
    
    fl.server.start_server(
        server_address=ip_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()