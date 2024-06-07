import copy
import logging
import math
import os
import time
import yaml
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda import amp  # Mixed precision training

from utils.dataloaders import create_dataloader
from model.yolov5.models.yolo import Model as YOLOV5
from model.yolov5.utils.general import intersect_dicts
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from model.yolov5.val import run as run_val
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

try:
    import wandb
except ImportError:
    wandb = None
    logging.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

# Load configuration from YAML file
with open("path.yaml", "r") as f:
    paths = yaml.safe_load(f)
    
def get_model_parameters(model):
    """Return a model's parameters."""
    logging.info("Fetching model parameters")
    parameters = [copy.deepcopy(val.cpu().numpy()) for _, val in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(parameters)

def get_weights(model):
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    """Set model weights from a list of NumPy ndarrays."""
    model.load_state_dict(weights, strict=True)

def load_model(args, device):
    """Load YOLOv5 model with pre-trained weights."""
    model = YOLOV5(cfg=paths["yolov5_cfg"])
    pretrained_weights_path = paths["pretrained_weights"]
    pretrained_dict = torch.load(pretrained_weights_path)["model"].state_dict()
    model_state_dict = model.state_dict()
    common_state_dict = intersect_dicts(model_state_dict, pretrained_dict)
    model.load_state_dict(common_state_dict, strict=False)

    yolo_hyp = paths["yolo_hyp"]
    with open(yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    hyp["cls"] *= 80 / 80.0
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])

    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {args.data}"
    args.nc = nc

    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.names = names
    return model

def load_val_dataloader(val_path, imgsz, batch_size, gs, single_cls, hyp, cache, rect, rank, workers, pad, prefix):
    """Load validation data loader."""
    logging.debug("Loading validation data loader...")
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
    logging.debug("Validation data loader loaded successfully.")
    return val_loader

def load_hyp(hyp_path):
    """Load hyperparameters from a YAML file."""
    hyp = check_yaml(hyp_path)
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    return hyp

def val(model, train_data, device, args, hyp):
    """Validate the model."""
    logging.info("Starting validation")
    hyp = hyp if hyp else args.hyp
    model.eval().to(device)
    compute_loss = ComputeLoss(model)

    t = time.time()
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

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
    logging.info(
        f"Validation results - Time: {time.time() - t:.2f}s, "
        f"Speed: {(time.time() - t) / len(train_data):.2f}s/batch, "
        f"Box: {box_loss:.4f}, Obj: {obj_loss:.4f}, Cls: {cls_loss:.4f}, Total: {box_loss + obj_loss + cls_loss:.4f}, "
        f"mAP: {map:.4f}, mAP50: {map50:.4f}, MP: {mp:.4f}, MR: {mr:.4f}"
    )
    return results


def train(train_data, model, device, args, hyp):
    try:
        logging.info(f"Start training on Trainer {args.id}")
        logging.info(f"Hyperparameters: {hyp}, Args: {args}")

        model = model.to(device).train()
        hyp = hyp if hyp else args.hyp
        epochs = args.epochs

        # Mixed precision training
        scaler = amp.GradScaler()

        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        if args.client_optimizer == "adam":
            optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))
        else:
            optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})
        optimizer.add_param_group({"params": pg2})
        logging.info(f"Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other")
        del pg0, pg1, pg2

        # Freeze
        freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                logging.info(f"Freezing {k}")
                v.requires_grad = False

        total_epochs = epochs * args.comm_round
        lf = lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        compute_loss = ComputeLoss(model)
        epoch_loss = []
        mloss = torch.zeros(3, device=device)

        for epoch in range(args.epochs):
            model.train()
            t = time.time()
            batch_loss = []
            logging.info(f"Epoch {epoch}/{epochs-1}")

            for batch_idx, batch in enumerate(train_data):
                imgs, targets, paths, _ = batch
                imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5

                with amp.autocast():
                    pred = model(imgs)
                    loss, loss_items = compute_loss(pred, targets.to(device).float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                batch_loss.append(loss.item())

                mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)
                mem = f"{torch.cuda.memory_reserved() / 1e9:.3g}G"

            scheduler.step()
            epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
            logging.info(f"Trainer epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {(time.time() - t)}")

            logging.info("#" * 20)
            logging.info(f"Trainer epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch")
            logging.info("#" * 20)

            if (epoch + 1) % args.checkpoint_interval == 0:
                model_path = args.save_dir / "weights" / f"model_client__epoch_{epoch}.pt"
                logging.info(f"Trainer epoch {epoch} saving model to {model_path}")
                torch.save(model.state_dict(), model_path)

            if (epoch + 1) % args.frequency_of_the_test == 0:
                logging.info(f"Start val on Trainer {args.id}")
                val(model=model, train_data=train_data, device=device, args=args, hyp=hyp)

        logging.info(f"End training on Trainer {args.id}")
        torch.save(model.state_dict(), args.save_dir / "weights" / f"model_client_.pt")

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
        traceback.print_exc()
        raise e

def load_model_init(args, device="cpu"):
    # Load configuration from path.yaml
    args.yolo_hyp = paths.get("yolo_hyp", "hyp.scratch.yaml")
    args.data_conf, args.yolo_cfg = (
        check_file(args.data_conf),
        check_file(args.yolo_cfg),
    )
    assert len(args.yolo_cfg) or len(args.weights), "either yolo_cfg or weights must be specified"
    args.img_size.extend([args.img_size[-1]] * (2 - len(args.img_size)))  # extend to 2 sizes (train, test)
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run

    # Load checkpoint interval
    args.checkpoint_interval = args.checkpoint_interval or 50
    args.server_checkpoint_interval = args.server_checkpoint_interval or 5

    # Load hyperparameters
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
        if "box" not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120"))
            hyp["box"] = hyp.pop("giou")

    args.total_batch_size = args.batch_size
    logging.info(f"Hyperparameters {hyp}")

    # Set up directories
    wdir = args.save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    args.last = wdir / "last.pt"
    args.best = wdir / "best.pt"
    args.results_file = args.save_dir / "results.txt"

    # Add file handler
    fh = logging.FileHandler(os.path.join(args.save_dir, f"log_{args.process_id}.txt"))
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    # Load data configuration
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    nc, names = (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (len(names), nc, args.data)
    args.nc = nc

    # Load model
    if args.model.lower() == "yolov5":
        pretrained = args.weights.endswith(".pt")
        if pretrained:
            ckpt = torch.load(args.weights, map_location=device)
            if hyp.get("anchors"):
                ckpt["model"].yaml["anchors"] = round(hyp["anchors"])
            model = YOLOv5(args.yolo_cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)
            exclude = ["anchor"] if args.yolo_cfg or hyp.get("anchors") else []
            state_dict = ckpt["model"].float().state_dict()
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
            model.load_state_dict(state_dict, strict=False)
            logging.info("Transferred %g/%g items from %s" % (len(state_dict), len(model.state_dict()), args.weights))
        else:
            model = YOLOv5(args.yolo_cfg, ch=3, nc=nc).to(device)

    args.model_stride = model.stride
    gs = int(max(model.stride))
    imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size]

    hyp["cls"] *= nc / 80.0
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.names = names

    # Set up optimizer
    nbs = 64
    accumulate = max(round(nbs / args.total_batch_size), 1)
    hyp["weight_decay"] *= args.total_batch_size * accumulate / nbs

    # Save settings
    with open(args.save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)

    args.hyp = hyp
    args.wandb = wandb

    return model

def fit_config(server_round: int):
    return {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }

def get_evaluate_fn(model, opt, model_weights_path, yolov5_weights):
    def evaluate(server_round: int, parameters, config: dict):
        try:
            params_dict_with_keys = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict_with_keys})
            model.load_state_dict(state_dict, strict=True)

            torch.save(model.state_dict(), model_weights_path)
            logging.info(f"Model weights saved to {model_weights_path}")

            loss = 0
            accuracy = {}
            data_dict = check_dataset(opt.data)
            train_path, val_path = data_dict.get("train"), data_dict.get("val")
            nc = 1 if opt.single_cls else int(data_dict.get("nc", 1))

            gs = max(int(model.stride.max()), 32)
            imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

            if RANK in {-1, 0}:
                results, _, _ = val.run(
                    data_dict,
                    batch_size=opt.batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    weights=yolov5_weights,
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