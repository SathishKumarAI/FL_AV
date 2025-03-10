import fedml
# from fedml import FedMLRunner
import Runner
from model.init_yolo import init_yolo
from trainer.yolo_aggregator import YOLOAggregator
import flwr as fl
import torch

if __name__ == "__main__":
    # init FedML framework
    # args = fedml.init()
    args = {
        'yolo_hyp': '',
        'data_conf': '',
        'yolo_cfg': '',
        'img_size': '',
        'save_dir': '',
        'checkpoint_interval': '',
        'server_checkpoint_interval': '',
        'total_batch_size': '',
        'epochs': '',
        'batch_size': '',
        'weights': '',
        'model': '',
    }


    print(args)
    print("aafaavasvsv")
    # args = fl.init()
    # args = 
    # init device
    # device = fedml.device.get_device(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # init yolo
    model, dataset, trainer, args = init_yolo(args=args, device=device)
    
    aggregator = YOLOAggregator(model, args)

    # start training
    fedml_runner = Runner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()




