import flwr as fl
import torch
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from flwr.common import (
    Code,
    ndarrays_to_parameters,
)
import train
import util
import val
from utils.general import (
    check_img_size,
    check_dataset,
)
import warnings
# Set the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
from model.init_yolo import init_yolo
warnings.filterwarnings("ignore", category=UserWarning)

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }

def get_evaluate_fn(model, opt):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters, config: dict):
        try:
            # Update model with the latest parameters
            params_dict_with_keys = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict_with_keys})
            model.load_state_dict(state_dict, strict=True)
            
            # Save model weights to .pt file
            model_weights_path = "/home/siu856522160/major/test/yolov5/model_weights.pt"
            torch.save(model.state_dict(), model_weights_path)
            logging.info(f"Model weights saved to {model_weights_path}")
            
            # Initialize variables
            loss = 0
            accuracy = {}
            data_dict = check_dataset(opt.data)
            train_path, val_path = data_dict.get("train"), data_dict.get("val")
            nc = 1 if opt.single_cls else int(data_dict.get("nc", 1))
            
            # Image size
            gs = max(int(model.stride.max()), 32)
            imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
            
            if RANK in {-1, 0}:
                results, _, _ = val.run(
                    data_dict,
                    batch_size=opt.batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    weights="/home/siu856522160/major/test/yolov5/yolov5s.pt",
                    iou_thres=0.60,
                    single_cls=opt.single_cls,
                    verbose=True,
                    plots=False,
                )
                keys = ['mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls']
                result_dict = {k: v for k, v in zip(keys, results)}
                logging.info(f"Evaluation results: {result_dict}")
                return loss, result_dict
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            return None, None

    return evaluate

def main():
    args = util.fit_config_file(r"major\config\config.yaml")
    args = SimpleNamespace(**args)
    IP_ADDRESS = "10.100.192.219:8080"
    
    # model = util.load_model(DEVICE)
    opt = train.parse_opt(True)
    model, dataset_train, dataset_val, args = init_yolo(args=args, device="cuda:0")
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        initial_parameters=ndarrays_to_parameters(model_parameters),
        
    )
    fl.server.start_server(
        server_address=IP_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
