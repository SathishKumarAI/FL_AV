import warnings
from collections import OrderedDict
import copy
import logging
import math
import time

import yaml

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import flwr as fl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# Import the parse_opt and main functions from detect.py
from detect import parse_opt, main
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.general import non_max_suppression, xywh2xyxy, scale_coords
from model.yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from model.yolov5.val import run as run_val
from model.yolov5.models.yolo import Model

import torch
from pathlib import Path
from models.yolo import Model 
from utils.general import check_img_size

args = {
        'yolo_hyp': r'..\\data\hyps\hyp.scratch.yaml',
        'data_conf': r'config/config.yaml',
        'yolo_cfg': '',
        'img_size': [640,640],
        'save_dir': '',
        'checkpoint_interval': '',
        'server_checkpoint_interval': '',
        'total_batch_size': '',
        'epochs': '',
        'batch_size': '',
        'weights': '',
        # 'model': '',
        "nc":10,
    }

def init_yolov5_model(args, device="cpu"):
    # Load the YOLOv5 model
    weights = r'.\\yolov5_test\runs\train\exp8\weights\best.pt'  # specify the path to your weights file
    model = Model(args["yolo_cfg"], ch=3, nc=args["nc"]).to(device)  # create the YOLOv5 model

    # Load pretrained weights if available
    if weights.endswith(".pt"):
        ckpt = torch.load(weights, map_location=device)
        state_dict = ckpt["model"].float().state_dict()
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
        model.load_state_dict(state_dict, strict=False)

    # Configure image size
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size]  # verify imgsz are grid size multiples

    return model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_yolov5_model(args, device=device)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, args=None):
        super(FlowerClient, self).__init__(model, args)
        self.hyp = args.hyp
        self.args = args
        self.round_loss = []
        self.round_idx = 0

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info(f"Start training on Trainer {self.id}")
        logging.info(f"Hyperparameters: {self.hyp}, Args: {self.args}")

        model = self.model
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp

        epochs = args.epochs

        optimizer = self._initialize_optimizer(model, hyp, args)

        # Freeze certain layers
        self._freeze_layers(model)

        total_epochs = epochs * args.comm_round

        scheduler = self._initialize_scheduler(optimizer, hyp, total_epochs)

        model.to(device)
        model.train()

        compute_loss = ComputeLoss(model)

        epoch_loss = []
        mloss = torch.zeros(3, device=device)  # mean losses
        logging.info("Epoch gpu_mem box obj cls total targets img_size time")

        for epoch in range(args.epochs):
            model.train()
            t = time.time()
            batch_loss = []
            logging.info(f"Trainer_ID: {self.id}, Epoch: {epoch}")

            for batch_idx, batch in enumerate(train_data):
                imgs, targets, paths, _ = batch
                imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5

                optimizer.zero_grad()
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device).float())  # loss scaled by batch_size

                # Backward
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses
                mem = f"%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                logging.info(
                    "%10s" * 2 + "%10.4g" * 5
                    % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                )

            scheduler.step()

            epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
            logging.info(
                f"Trainer {self.id} epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {time.time() - t}"
            )

            logging.info("#" * 20)

            logging.info(
                f"Trainer {self.id} epoch {epoch} time: {time.time() - t}s batch_num: {batch_idx} speed: {(time.time() - t) / batch_idx} s/batch"
            )
            logging.info("#" * 20)

            if (epoch + 1) % self.args.checkpoint_interval == 0:
                model_path = self.args.save_dir / "weights" / f"model_client_{self.id}_epoch_{epoch}.pt"
                torch.save(model.state_dict(), model_path)
                logging.info(f"Trainer {self.id} epoch {epoch} saving model to {model_path}")

            if (epoch + 1) % self.args.frequency_of_the_test == 0:
                logging.info(f"Start val on Trainer {self.id}")
                self.val(model, train_data, device, args)

        logging.info(f"End training on Trainer {self.id}")
        torch.save(model.state_dict(), self.args.save_dir / "weights" / f"model_client_{self.id}.pt")

        # plot for client
        # plot box, obj, cls, total loss
        epoch_loss = np.array(epoch_loss)
        # logging.info(f"Epoch loss: {epoch_loss}")
        # Import the required libraries
        

        # Log the training metrics using flwr.client_manager.ClientManager.log
        metrics = {
            "round_idx": self.round_idx,
            "train_box_loss": float(epoch_loss[-1, 0]),
            "train_obj_loss": float(epoch_loss[-1, 1]),
            "train_cls_loss": float(epoch_loss[-1, 2]),
            "train_total_loss": float(epoch_loss[-1, :].sum())
        }
        # Log the metrics using flwr.client_manager.ClientManager.log
        self.client_manager.log(metrics)

        self.round_loss.append(epoch_loss[-1, :])
        if self.round_idx == args.comm_round:
            self.round_loss = np.array(self.round_loss)
            # logging.info(f"round_loss shape: {self.round_loss.shape}")
            logging.info(
                f"Trainer {self.id} round {self.round_idx} finished, round loss: {self.round_loss}"
            )
        return


    def val(self, model, train_data, device, args):
        
        logging.info(f"Trainer {self.id} val start")
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp

        model.eval()
        model.to(device)
        compute_loss = ComputeLoss(model)

        t = time.time()
        with open(args.data_conf, 'r') as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

        results, maps, _ = run_val(
            data_dict,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            model=model,
            single_cls=args.single_cls,
            dataloader=train_data,
            save_dir=args.save_dir,
            plots=False,
            compute_loss=compute_loss,
        )
        logging.info(results)
        mp, mr, map50, map, box_loss, obj_loss, cls_loss = results
        logging.info(f"Trainer {self.id} val time: {time.time() - t}s speed: {(time.time() - t) / len(train_data)} s/batch")
        logging.info(f"Trainer {self.id} val box: {box_loss} obj: {obj_loss} cls: {cls_loss} total: {box_loss + obj_loss + cls_loss}")
        logging.info(f"Trainer {self.id} val map: {map} map50: {map50} mp: {mp} mr: {mr}")

        return



# # Start Flower client
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=FlowerClient(),
# )
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model))
