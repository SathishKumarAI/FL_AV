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
import temp.major.util as util
import warnings
import val
import temp.major.flower.val_copy_test as val_copy_test
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
class FlowerClient(fl.client.Client):
    def __init__(self, device, args, model, opt, num_examples):
        super().__init__()
        self.device = device
        self.args = args
        self.model = model
        self.opt = opt
        self.num_examples = num_examples
        logging.info("init_model_params")

    def set_parameters(self, parameters):
        """Set model parameters."""
        try:
            # Update local model parameters
            logging.info("Setting model parameters")
            t = parameters_to_ndarrays(parameters.parameters)
            params_dict_with_keys = zip(list(self.model.state_dict().keys()), t)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict_with_keys})

            # Load state dictionary into model
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logging.error(f"An error occurred during setting parameters: {e}")

    def fit(self, parameters):
        """Train parameters on the locally held training set."""
        try:
            logging.info("Fitting model parameters")
            self.set_parameters(parameters)
            # Get hyperparameters for this round
            batch_size = 16
            results = train.run(
                data=self.args.data_conf,
                imgsz=1024,
                weights='/home/siu856522160/major/test/yolov5/yolov5s.pt',
                batch=batch_size,
                epochs=2,
                device=self.device
            )
            keys = ['mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls']
            result_dict = {k: v for k, v in zip(keys, results)}
            parameters = util.get_model_parameters(self.model)
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
            current_model_state_dict = self.model.state_dict()
            loss = 0
            accuracy = {}
            data_dict = check_dataset(self.opt.data)
            if data_dict is None:
                logging.error("Dataset information not found.")
                return EvaluateRes(
                    status=Status(Code.EVALUATE_NOT_IMPLEMENTED, message="ok"),
                    loss=loss,
                    num_examples=self.num_examples.get("testset", 0),
                    metrics={},
                )
            
            # Print the dataset information
            logging.info(f"Dataset information: {data_dict}")

            train_path = data_dict.get("train")
            val_path = data_dict.get("val")
            if train_path is None:
                logging.error("Train path not found in data_dict.")
            if val_path is None:
                logging.error("Validation path not found in data_dict.")
            names = data_dict.get("names")
            if not names:
                logging.error("Names not available in data_dict.")

            gs = max(int(self.model.stride.max()), 32)
            imgsz = check_img_size(self.opt.imgsz, gs, floor=gs * 2)
            amp = check_amp(self.model)  # check AMP

            # Estimate batch size
            batch_size = self.opt.batch_size
            if batch_size == -1:  # single-GPU only, estimate best batch size
                batch_size = check_train_batch_size(self.model, imgsz, amp)

            single_cls = self.opt.single_cls
            hyp_path = "/home/siu856522160/major/test/tt/yolov5/config/hyps/hyp.scratch-low.yaml"
            hyp = util.load_hyp(hyp_path)
            cache = False
            rect = False
            rank = 0
            workers = 4
            pad = 0.5
            prefix = colorstr("val: ")

            # Load validation data loader
            val_loader = util.load_val_dataloader(val_path, imgsz, batch_size, gs, single_cls, hyp, cache, rect, rank, workers, pad, prefix)
            logging.info("Validation data loader:", val_loader)

            # Run validation
            results, _, _ = val_copy_test.run(
                data=data_dict,
                imgsz=imgsz,
                model=self.model,
                iou_thres=0.60,
                single_cls=self.opt.single_cls,
                verbose=True,
                plots=False,
            )
            logging.info("Evaluating results model parameters")

            # Debugging statement
            logging.info("Num Examples:", self.num_examples)

            # Restore the model state
            self.model.load_state_dict(current_model_state_dict)

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
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. Picks partition 0 by default",
    )
    args = parser.parse_args()
    yaml_path = r"/home/siu856522160/major/test/tt/yolov5/config/config.yaml"
    model_path = r"/home/siu856522160/major/test/yolov5/yolov5s.pt"
    args = util.fit_config_file(yaml_path)
    args = SimpleNamespace(**args)
    IP_ADDRESS = "10.100.192.219:8080"
    num_examples = {"trainset" : 80, "testset" : 80}
    opt = train.parse_opt(True)
    model = util.load_model(DEVICE)
    client = FlowerClient(device=DEVICE, args=args, model=model, opt=opt, num_examples=num_examples )
    fl.client.start_client(server_address=IP_ADDRESS, client=client)

if __name__ == "__main__":
    main()
