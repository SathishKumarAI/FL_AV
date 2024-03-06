import flwr as fl
from model.init_yolo import init_yolo
from trainer.yolo_aggregator import YOLOAggregator
import os
import sys
from pathlib import Path
# import fedml
# from fedml import FedMLRunner
from model.init_yolo import init_yolo
from trainer.yolo_aggregator import YOLOAggregator
import torch
from pathlib import PosixPath
from torch import tensor, device
from types import SimpleNamespace
# from trainer.yolov5_trainer import train, val
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class YOLOFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

    # def fit(self, parameters, config):
    #     self.model.load_state_dict(parameters)
    #     # train(self, train_data, device, args):
    #     train()
    #     # Implement your training logic here
    #     # You can use self.model an   d self.args to access your YOLO model and training arguments
    #     # Return a dictionary containing the updated model parameters and the number of examples processed

    # def evaluate(self, parameters, config):
    #     self.model.load_state_dict(parameters)
    #     # val(self, model, train_data, device, args)
    #     # Implement your evaluation logic here
    #     # You can use self.model and self.args to access your YOLO model and training arguments
    #     # Return a dictionary containing evaluation metrics

if __name__ == "__main__":


    model, dataset, _, args = init_yolo(args=args, device="cuda:0")  # Pass appropriate args and device
    # print(model)
    print(dataset)

    # aggregator = YOLOAggregator(model, args)
    # aggregator = YOLOAggregator()
    # Start Flower server
    fl.server.start_server(
        # config={"num_rounds": 100}, 
                           strategy=fl.server.strategy.FedAvg(), 
                        #    address="127.0.0.1:8080"
                           )

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=YOLOFlowerClient(model, args))
