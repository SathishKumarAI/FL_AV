#!/usr/bin/env bash
RANK=$1
python main.py --cf config/config_yolov5.yaml --run_id yolov5 --rank $RANK --role client