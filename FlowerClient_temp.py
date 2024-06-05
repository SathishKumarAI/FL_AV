import flwr as fl
import torch
import yaml
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from types import SimpleNamespace
from collections import OrderedDict
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import train
import util
import warnings
import val
import val_copy_test
import numpy as np
import argparse
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_img_size,
    check_dataset,
    colorstr,
)
import traceback
from utils.autobatch import check_train_batch_size
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
warnings.filterwarnings("ignore", category=UserWarning)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,)


from model.init_yolo import init_yolo
import copy
from model.yolov5.models.yolo import Model as YOLOV5

class FlowerClient1(fl.client.Client):
    def __init__(self, device, args, model, opt, num_examples, train_data, val_data):
        super().__init__()
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.opt = opt
        self.num_examples = num_examples
        self.train_data = train_data
        self.val_data = val_data
        logging.info("init_model_params")

    def get_model_parameters(self):
        """Return a model's parameters."""
        logging.info("Getting model parameters")
        parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return fl.common.ndarrays_to_parameters(parameters)

    def set_parameters(self, parameters):
        """Set model parameters."""
        try:
            # Update local model parameters with logging
            logging.info("Setting model parameters")
            state_dict = {}
            model_state_dict = self.model.state_dict()
            t = parameters_to_ndarrays(parameters.parameters)
            # Ensure no gradient computation if not required
            with torch.no_grad():
                for (name, param), value in zip(model_state_dict.items(), t):
                    param_shape = param.shape
                    # Convert and check parameter shape mismatch
                    param_value = torch.tensor(value)
                    if param_shape != param_value.shape:
                        logging.error(f"Shape mismatch for {name}: {param_value.shape} vs {param_shape}. Skipping...")
                        continue
                    state_dict[name] = param_value
            # Load state dictionary with strict=False (consider implications)
            # Create a new instance of the model
            new_model = util.load_model_init(self.args, self.device)
            # Load the state dictionary
            new_model.load_state_dict(state_dict, strict=False)
            # Replace the old model with the new one
            self.model = new_model
            # self.model.load_state_dict(state_dict, strict=False)
            # self.model = model_1 
            logging.info("Model parameters set successfully.")
        except Exception as e:
            logging.error(f"An error occurred during setting parameters: {e}")

    def fit(self, parameters):
        """Train parameters on the locally held training set."""
        try:
            logging.info("Fitting model parameters")
            self.set_parameters(parameters)
            # Set the model to training mode
            # self.model.train()        
            # Get hyperparameters for this round
            hyp_path = "/home/siu856522160/major/test/tt/yolov5/config/hyps/hyp.scratch-low.yaml"
            hyp = util.load_hyp(hyp_path)
            results = train.run(        
                                data=self.args.data_conf,
                                imgsz=1024,
                                weights='/home/siu856522160/major/test/yolov5/yolov5s.pt',
                                batch=32,
                                epochs=2,
                                device=self.device            )
            keys = ["train_box_loss", "train_obj_loss", "train_cls_loss", "train_total_loss"]
            result_dict = {k: v for k, v in zip(keys, results)}
            parameters = self.get_model_parameters()
            logging.info("Fitting model returns")
            return FitRes(
                status=Status(Code.OK, message="ok"),
                parameters=parameters,
                num_examples=self.num_examples.get("trainset", 0),
                metrics=result_dict
            )
        except Exception as e:
            logging.error(f"An error occurred during fitting: {e}")
            return FitRes(
                status=Status(Code.FIT_NOT_IMPLEMENTED, message="ok"),
                parameters=parameters,
                num_examples=self.num_examples.get("trainset", 0),
                metrics={}
            )

    def evaluate(self, parameters):
        """Evaluate parameters on the locally held test set."""
        try:
            logging.info("Evaluating model parameters")
            self.set_parameters(parameters)
            loss = 0
            accuracy = {}



            hyp_path = "/home/siu856522160/major/test/tt/yolov5/config/hyps/hyp.scratch-low.yaml"
            hyp = util.load_hyp(hyp_path)

            
            # Run validation
            results = util.val(
                model=self.model, 
                train_data=self.val_data,
                device=self.device,
                args=self.args,
                hyp=hyp
            )
            logging.info("Evaluating results model parameters")

            # Debugging statement
            logging.info("Num Examples:", self.num_examples)

            # Create result dictionary
            keys = ['mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls']
            result_dict = {k: v for k, v in zip(keys, results)}

            return EvaluateRes(
                status=Status(Code.OK, message="ok"),
                loss=loss,
                num_examples=self.num_examples.get("testset", 0),
                metrics=result_dict,
            )

        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            traceback.print_exc()  # Print the traceback information
            return EvaluateRes(
                status=Status(Code.EVALUATE_NOT_IMPLEMENTED, message="ok"),
                loss=loss,
                num_examples=self.num_examples.get("testset", 0),
                metrics={},
            )

def main():
    parser = argparse.ArgumentParser(description="Process and save tid from command line.")
    parser.add_argument("--id", type=int, help="The tid to be saved")

    args = parser.parse_args()

    # Accessing the value of tid
    id = args.id

    # Save the tid or perform any other operation
    # print("TID:", id)
    # yaml_path = r"/home/siu856522160/major/test/tt/yolov5/config/config.yaml"
    yaml_path = f'/home/siu856522160/major/test/tt/yolov5/config/config_copy_{id}.yaml'
    model_path = f"/home/siu856522160/major/test/tt/yolov5/yolov5s.pt"
    args = util.fit_config_file(yaml_path)
    args = SimpleNamespace(**args)
    IP_ADDRESS = "10.100.192.219:8080"
    num_examples = {"trainset": 80, "testset": 80}
    opt = train.parse_opt(True)
    model, dataset_train, dataset_val, args = init_yolo(args=args, device="cuda:0")
    index = 1
    dataset_train = dataset_train[index][0]
    dataset_val = dataset_val[index][0]
    
    # print(dataset_train)
    client = FlowerClient1(device=DEVICE, args=args, model=model, num_examples=num_examples, opt=opt, train_data=dataset_train, val_data=dataset_val)
    fl.client.start_client(server_address=IP_ADDRESS, client=client)

if __name__ == "__main__":
    main()