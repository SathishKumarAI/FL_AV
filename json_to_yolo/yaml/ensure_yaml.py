import os
import yaml
import logging
import torch
from collections import OrderedDict
from typing import List, Optional

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "process.log")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
    print("Logging initialized. Check logs/process.log for details.")
    logging.info("Logging setup complete.")
setup_logging()

def ensure_data_yaml_exists(batch_root):
    logging.info("Ensuring data.yaml exists for all batches...")
    if not os.path.exists(batch_root):
        logging.error(f"Batch root directory {batch_root} does not exist.")
        return
    
    for batch in os.listdir(batch_root):
        logging.info(f"Processing batch: {batch}")
        batch_path = os.path.join(batch_root, batch)
        data_yaml_path = os.path.join(batch_path, 'data.yaml')
        if os.path.isdir(batch_path):
            if not os.path.exists(data_yaml_path):
                try:
                    logging.info(f"Creating missing data.yaml in {batch_path}...")
                    batch_number = batch.split('_')[-1]
                    category_mapping = {
                        "person": 1, "pedestrian": 1, "rider": 2, "car": 3, "truck": 4,
                        "bus": 5, "train": 6, "motor": 7, "motorcycle": 7, "bike": 8,
                        "bicycle": 8, "traffic light": 9, "traffic sign": 10, "trailer": 11,
                        "other person": 12, "other vehicle": 13
                    }
                    category_names = {v: k for k, v in category_mapping.items()}
                    data_yaml_content = {
                        'path': os.path.abspath(batch_path),
                        'train': os.path.join(batch_path, 'train.txt'),
                        'val': os.path.join(batch_path, 'val.txt'),
                        'test': os.path.join(batch_path, 'test.txt'),
                        'names': category_names
                    }
                    with open(data_yaml_path, 'w') as f:
                        yaml.dump(data_yaml_content, f, default_flow_style=False)
                    logging.info(f"data.yaml created successfully in {batch_path}.")
                except Exception as e:
                    logging.error(f"Error creating data.yaml in {batch_path}: {e}")
            else:
                logging.info(f"data.yaml already exists in {batch_path}")

def create_splits(batch_root):
    logging.info("Creating dataset split files (train.txt, val.txt, test.txt) for all batches...")
    if not os.path.exists(batch_root):
        logging.error(f"Batch root directory {batch_root} does not exist.")
        return
    
    for batch in os.listdir(batch_root):
        logging.info(f"Processing batch: {batch}")
        batch_path = os.path.join(batch_root, batch)
        if os.path.isdir(batch_path):
            train_file = os.path.join(batch_path, "train.txt")
            val_file = os.path.join(batch_path, "val.txt")
            test_file = os.path.join(batch_path, "test.txt")

            train_list, val_list, test_list = [], [], []
            
            train_dir = os.path.join(batch_path, 'train', 'images')
            val_dir = os.path.join(batch_path, 'val', 'images')
            test_dir = os.path.join(batch_path, 'test', 'images')
            
            if os.path.exists(train_dir):
                train_list.extend([os.path.join(batch_path, 'train', 'images', f) for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))])
            if os.path.exists(val_dir):
                val_list.extend([os.path.join(batch_path, 'val', 'images', f) for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))])
            if os.path.exists(test_dir):
                test_list.extend([os.path.join(batch_path, 'test', 'images', f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])
            
            try:
                with open(train_file, 'w') as f:
                    f.write('\n'.join(train_list) + '\n')
                with open(val_file, 'w') as f:
                    f.write('\n'.join(val_list) + '\n')
                with open(test_file, 'w') as f:
                    f.write('\n'.join(test_list) + '\n')
                logging.info(f"Dataset split files created successfully for {batch}.")
            except Exception as e:
                logging.error(f"Error writing dataset split files for {batch}: {e}")

def process_batches(batch_root):
    logging.info("Starting batch processing... Logging each task execution...")
    ensure_data_yaml_exists(batch_root)
    create_splits(batch_root)
    logging.info("Batch processing completed successfully.")
    print("Batch processing completed. Check logs/process.log for details.")

if __name__ == "__main__":
    batch_root = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\bdd100_batch"  # Update this path if necessary
    process_batches(batch_root)
