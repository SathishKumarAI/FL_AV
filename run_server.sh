# #!/usr/bin/env bash
# python main_fedml_object_detection.py --cf config/fedml_config.yaml --run_id yolov5 --rank 0 --role server
#!/usr/bin/env bash
python main.py --cf config/config.yaml --run_id yolov5 --rank 0 --role server --ip 0.0.0.0:8080