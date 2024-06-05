import flwr as fl
import torch
import yaml
import logging
from pathlib import Path
import warnings
import train
from collections import OrderedDict
from models.experimental import attempt_load
from types import SimpleNamespace
import util
from typing import Dict, List, Tuple
import numpy as np
import argparse
# Set the device
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)
class FlowerClient(fl.client.Client):
    def __init__(self, device, args, model, opt, num_examples):
        self.device = device
        self.args = args
        self.model = model
        self.opt = opt
        self.num_examples = num_examples
        logging.info("init_model_params")
    # def get_parameters(self, config) :
    #     # Return model parameters as a list of NumPy ndarrays
    #     logging.info("get_model_params")
    #     return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        """Set model parameters."""
        # Update local model parameters
        logging.info("set_model_params")    
        # print(type(parameters.parameters.tensors))
        # print(type(parameters.parameters.tensor_type))
        t = fl.common.parameters_to_ndarrays(parameters.parameters)
        # Create state dictionary 
        # print(t[1])
        # tensor_data = torch.tensor(t,dtype=torch.float32)
        params_dict_with_keys = zip(list(self.model.state_dict().keys()), t)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict_with_keys})
        # Load state dictionary into model
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters):
        """Train parameters on the locally held training set."""
        # print(parameters)
        logging.info("fit_model_params")
        self.set_parameters(parameters)
        
        # Get hyperparameters for this round
        batch_size = 16
        # epochs: int = config["local_epochs"]
        
        # Execute training
        print(self.args.data_conf)
        results = train.run(
            data=self.args.data_conf, 
            imgsz=1024, 
            weights='/home/siu856522160/major/test/yolov5/yolov5s.pt',
            batch=batch_size, 
            epochs=2, 
            device=self.device
        )
        
        # print(results)
        # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), box, obj, cls
        
        # Convert lists to dictionary
        keys = ['mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls']
        result_dict = {k: v for k, v in zip(keys, results)}
        parameters = util.get_model_params(self.model)
        print(result_dict)
        num_examples = self.args.nc_index
        logging.info("fit_model_results")
        # Create and return FitRes
        return fl.common.FitRes(
            status=fl.common.Status(fl.common.Code.OK, message="ok"), 
            parameters=parameters, 
            num_examples=self.num_examples["trainset"], 
            metrics=result_dict
        )


    def evaluate(self, parameters   ) :
        """Evaluate parameters on the locally held test set."""
        self.set_parameters(parameters)
        # Execute evaluation
        loss = 0
        accuracy = {}
        
        # Create and return EvaluateRes
        return fl.common.EvaluateRes(status=fl.common.Status(fl.common.Code.OK,message="ok"),loss=loss, num_examples=self.num_examples["testset"], metrics=accuracy)

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    args = parser.parse_args()
    # Load configuration
    yaml_path = r"/home/siu856522160/major/test/tt/yolov5/config/config_yolov5.yaml"
    model_path = r"/home/siu856522160/major/test/yolov5/yolov5s.pt"
    args = util.fit_config_file(yaml_path)
    args = SimpleNamespace(**args)
    IP_ADDRESS = "10.100.192.219:8080"
    num_examples = {"trainset" : 80, "testset" : 80}
    
    
    # Start Flower client
    opt = train.parse_opt(True)
    model = util.load_model(DEVICE)
    print("dadaada")
    client = FlowerClient(device=DEVICE, args=args, model=model, opt=opt, num_examples=num_examples ).to_client()
    fl.client.start_client(server_address=IP_ADDRESS, client=client)

if __name__ == "__main__":
    main()
