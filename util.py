import logging
import yaml
import os
import flwr as fl
import torch
from pathlib import Path
from model.yolov5.models.yolo import Model as YOLOV5
from model.yolov5.utils.general import intersect_dicts
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
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


from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.general import non_max_suppression, xywh2xyxy, scale_coords
from model.yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from model.yolov5.val import run as run_val

def fit_config_file(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)
import copy

import logging
import traceback


import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from warnings import warn
import yaml
import torch

from data.data_loader import load_partition_data_coco, load_entire_training_data_coco, load_entire_validation_data_coco
from model.yolov5.utils.general import (
    labels_to_class_weights,
    increment_path,
    check_file,
    check_img_size,
)
from model.yolov5.utils.general import intersect_dicts
from model.yolov5.models.yolo import Model as YOLOv5


from trainer.yolov5_trainer import YOLOv5Trainer


try:
    import wandb
except ImportError:
    wandb = None
    logging.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )


def get_model_parameters(model):
    """Return a model's parameters."""
    logging.info("get_model_parameters")
    # Instead of directly converting parameters to numpy arrays
    parameters = [copy.deepcopy(val.cpu().numpy()) for _, val in model.state_dict().items()]

    # Convert parameters to Parameters object
    return fl.common.ndarrays_to_parameters(parameters)
    # return fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])

def get_weights(model):
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    """Set model weights from a list of NumPy ndarrays."""
    model.load_state_dict(weights, strict=True)

def load_model(args,device):
    """Load YOLOv5 model with pre-trained weights."""
    model = YOLOV5(cfg="/home/siu856522160/major/test/yolov5/models/yolov5s.yaml")
    pretrained_weights_path = "/home/siu856522160/major/test/yolov5/yolov5s.pt"
    pretrained_dict = torch.load(pretrained_weights_path)["model"].state_dict()
    model_state_dict = model.state_dict()
    common_state_dict = intersect_dicts(model_state_dict, pretrained_dict)
    model.load_state_dict(common_state_dict, strict=False)
    # Hyperparameters
    yolo_hyp = "/home/siu856522160/major/test/tt/yolov5/config/hyps/hyp.scratch-low.yaml"
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        
    hyp["cls"] *= 80 / 80.0
    # Configure
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        args.data,
    )  # check
    args.nc = nc  # change nc to actual number of classes

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(train_data_global.dataset.labels, nc).to(
    # device
    # )  # attach class weights
    model.names = names
    return model

def load_val_dataloader(val_path, imgsz, batch_size, gs, single_cls, hyp, cache, rect, rank, workers, pad, prefix):
    """Load validation data loader."""
    print("Loading validation data loader...")  # Debugging statement
    val_loader = create_dataloader(
        path=val_path,
        imgsz=imgsz,
        batch_size=batch_size,
        stride=gs,
        hyp=hyp,
        rect=rect,
        rank=rank,
        pad=pad,
        prefix=prefix
    )[0]
    print("Validation data loader loaded successfully.")  # Debugging statement
    return val_loader

def load_hyp(hyp_path):
    hyp = check_yaml(hyp_path)
    
    # Hyperparameters
    with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    # print("in load hyp", hyp)
    return hyp
def val( model, train_data, device, args, hyp):
        logging.info(f"Trainer  val start")
        # round_idx = args.round_idx
        args = args
        hyp = hyp if hyp else args.hyp

        model.eval()
        model.to(device)
        compute_loss = ComputeLoss(model)

        # val
        t = time.time()
        with open(args.data_conf) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
            
        # index = 3
        # value = train_data[index][0]
        # # print(value)
        
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
        # print(results)
        logging.info(
            f"Trainer  val time: {(time.time() - t)}s speed: {(time.time() - t)/len(train_data)} s/batch"
        )
        logging.info(
            f"Trainer  val box: {box_loss} obj: {obj_loss} cls: {cls_loss} total: {box_loss + obj_loss + cls_loss}"
        )
        logging.info(
            f"Trainer  val map: {map} map50: {map50} mp: {mp} mr: {mr}"
        )

        return results


  

def train(train_data, model, device, args, hyp):
    try:
        id = args.id 
        logging.info("Start training on Trainer {}".format(id))
        logging.info(f"Hyperparameters: {hyp}, Args: {args}")
        model = model
        # round_idx = args.round_idx
        args = args
        hyp = hyp if hyp else args.hyp

        epochs = args.epochs  # number of epochs

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if args.client_optimizer == "adam":
            optimizer = optim.Adam(
                pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
            )  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(
                pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
            )

        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        logging.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )
        del pg0, pg1, pg2

        # Freeze
        freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            # v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                logging.info("Freezing %s" % k)  # Debugging statement
                v.requires_grad = False

        total_epochs = epochs * args.comm_round

        lf = (
            lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2)
            * (1 - hyp["lrf"])
            + hyp["lrf"]
        )  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

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
            logging.info("Trainer_ID: , Epoch: {0}".format(epoch))

            for (batch_idx, batch) in enumerate(train_data):
                imgs, targets, paths, _ = batch
                imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5

                optimizer.zero_grad()
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device).float()
                )  # loss scaled by batch_size

                # Backward
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                mloss = (mloss * batch_idx + loss_items) / (
                    batch_idx + 1
                )  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 5) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                logging.info(s)

            scheduler.step()

            epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
            logging.info(
                f"Trainer   epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {(time.time() - t)}"
            )

            logging.info("#" * 20)

            logging.info(
                f"Trainer  epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch"
            )
            logging.info("#" * 20)

            if (epoch + 1) % args.checkpoint_interval == 0:
                model_path = (
                    args.save_dir
                    / "weights"
                    / f"model_client__epoch_{epoch}.pt"
                )
                logging.info(
                    f"Trainer  epoch {epoch} saving model to {model_path}"
                )
                torch.save(model.state_dict(), model_path)

            if (epoch + 1) % args.frequency_of_the_test == 0:
                logging.info("Start val on Trainer {}".format(id))
                val(model=model, train_data=train_data, device=device, args=args, hyp=hyp)

        logging.info("End training on Trainer {}".format(id))
        torch.save(
            model.state_dict(),
            args.save_dir / "weights" / f"model_client_.pt",
        )

        epoch_loss = np.array(epoch_loss)
        fl.common.logger.configure({
                f"train_box_loss": float(epoch_loss[-1, 0]),
                f"train_obj_loss": float(epoch_loss[-1, 1]),
                f"train_cls_loss": float(epoch_loss[-1, 2]),
                f"train_total_loss": float(epoch_loss[-1, :].sum()),
        })

        return {
            float(epoch_loss[-1, 0]), 
            float(epoch_loss[-1, 1]), 
            float(epoch_loss[-1, 2]),
            float(epoch_loss[-1, :].sum())
        }

    except Exception as e:
        logging.error(f"Error occurred during training: {e}")
        traceback.print_exc()  # Print the traceback
        raise e

def load_model_init(args, device="cpu"):
    # print(type(args))
    # print(args.get("weights"))
    # init settings
    args.yolo_hyp = args.yolo_hyp or (
        "hyp.finetune.yaml" if args.weights else "hyp.scratch.yaml"
    )
    args.data_conf, args.yolo_cfg, args.yolo_hyp = (
        check_file(args.data_conf),
        check_file(args.yolo_cfg),
        check_file(args.yolo_hyp),
    )  # check files
    assert len(args.yolo_cfg) or len(
        args.weights
    ), "either yolo_cfg or weights must be specified"
    args.img_size.extend(
        [args.img_size[-1]] * (2 - len(args.img_size))
    )  # extend to 2 sizes (train, test)
    # args.name = "evolve" if args.evolve else args.name
    args.save_dir = increment_path(
        Path(args.project) / args.name, exist_ok=args.exist_ok
    )  # increment run

    # add checkpoint interval
    logging.info("add checkpoint interval")
    args.checkpoint_interval = (
        50 if args.checkpoint_interval is None else args.checkpoint_interval
    )
    args.server_checkpoint_interval = (
        5
        if args.server_checkpoint_interval is None
        else args.server_checkpoint_interval
    )

    # Hyperparameters
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    args.total_batch_size = args.batch_size

    logging.info(f"Hyperparameters {hyp}")
    save_dir, epochs, batch_size, total_batch_size, weights = (
        Path(args.save_dir),
        args.epochs,
        args.batch_size,
        args.total_batch_size,
        args.weights,
    )

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = save_dir / "results.txt"

    # add file handler
    logging.info("add file handler")
    fh = logging.FileHandler(os.path.join(args.save_dir, f"log_{args.process_id}.txt"))
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    args.last, args.best, args.results_file = last, best, results_file

    # Configure
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        args.data,
    )  # check
    args.nc = nc  # change nc to actual number of classes

    # Mode
    # print("weights:", weights)

    if args.model.lower() == "yolov5":
        pretrained = weights.endswith(".pt")
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            if hyp.get("anchors"):
                ckpt["model"].yaml["anchors"] = round(
                    hyp["anchors"]
                )  # force autoanchor
            model = YOLOv5(args.yolo_cfg or ckpt["model"].yaml, ch=3, nc=nc).to(
                device
            )  # create
            exclude = (
                ["anchor"] if args.yolo_cfg or hyp.get("anchors") else []
            )  # exclude keys
            state_dict = ckpt["model"].float().state_dict()  # to FP32
            state_dict = intersect_dicts(
                state_dict, model.state_dict(), exclude=exclude
            )  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            logging.info(
                "Transferred %g/%g items from %s"
                % (len(state_dict), len(model.state_dict()), weights)
            )  # report
        else:
            model = YOLOv5(args.yolo_cfg, ch=3, nc=nc).to(device)  # create


    
    args.model_stride = model.stride
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in args.img_size
    ]  # verify imgsz are gs-multiples

    hyp["cls"] *= nc / 80.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(train_data_global.dataset.labels, nc).to(
    # device
    # )  # attach class weights
    model.names = names
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    # logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    # with open(save_dir / "opt.yaml", "w") as f:
    #     # save args as yaml
    #     yaml.dump(args.__dict__, f, sort_keys=False)

    args.hyp = hyp  # add hyperparameters
    args.wandb = wandb



    # return model, dataset, dataset_train, dataset_val, trainer, args
    return model

