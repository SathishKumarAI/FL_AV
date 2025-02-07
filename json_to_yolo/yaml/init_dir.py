import os
import shutil
import yaml
import logging
import random

# ======================= SETUP LOGGING ===========================
def setup_logging():
    """Initializes logging for tracking dataset processing."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "process.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w"
    )
    print("Logging initialized. Check logs/process.log for details.")
    logging.info("Logging setup complete.")

setup_logging()

# ======================= CATEGORY MAPPING ========================
CATEGORY_MAPPING = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "train": 5, "motorcycle": 6, "bicycle": 7, "traffic light": 8, 
    "traffic sign": 9, "trailer": 10, "other person": 11, "other vehicle": 12
}

CATEGORY_NAMES = [name for name, index in sorted(CATEGORY_MAPPING.items(), key=lambda item: item[1])]

# ======================= INITIALIZE DATASET =======================
def initialize_dataset_directory(dataset_path):
    """Ensures the dataset directory has the necessary structure and files."""
    logging.info(f"Initializing dataset directory: {dataset_path}")
    
    os.makedirs(dataset_path, exist_ok=True)

    # Define necessary subdirectories
    subdirs = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
    for subdir in subdirs:
        os.makedirs(os.path.join(dataset_path, subdir), exist_ok=True)

    # Create/update data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    data_yaml_content = {
        "path": os.path.abspath(dataset_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CATEGORY_NAMES),
        "names": CATEGORY_NAMES
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    # Create empty train.txt, val.txt, test.txt if they don’t exist
    for split in ["train", "val", "test"]:
        split_file = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(split_file):
            with open(split_file, "w", encoding="utf-8") as f:
                f.write("")  # Create empty file
            logging.info(f"Created empty {split}.txt in {dataset_path}")

    logging.info(f"Dataset directory structure initialized in {dataset_path}.")

# ======================= CREATE MINI DATASET =======================
def create_mini_dataset(source_root, dest_root, num_images=10, num_batches=3):
    """Creates a mini dataset by randomly selecting images and copying labels."""
    if not os.path.exists(source_root):
        logging.error(f"Source directory '{source_root}' does not exist.")
        return

    os.makedirs(dest_root, exist_ok=True)

    for batch_num in range(1, num_batches + 1):
        source_batch = os.path.join(source_root, f"batch_{batch_num}")
        dest_batch = os.path.join(dest_root, f"batch_{batch_num}")

        if not os.path.exists(source_batch):
            logging.warning(f"Skipping batch {batch_num}, missing: {source_batch}")
            continue

        os.makedirs(dest_batch, exist_ok=True)

        for split in ["train", "val", "test"]:
            source_images_dir = os.path.join(source_batch, split, "images")
            source_labels_dir = os.path.join(source_batch, split, "labels")
            dest_images_dir = os.path.join(dest_batch, split, "images")
            dest_labels_dir = os.path.join(dest_batch, split, "labels")

            if not os.path.exists(source_images_dir):
                logging.warning(f"Skipping {split} in batch {batch_num}, missing: {source_images_dir}")
                continue

            os.makedirs(dest_images_dir, exist_ok=True)
            os.makedirs(dest_labels_dir, exist_ok=True)

            image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png'))]
            if not image_files:
                logging.warning(f"No images found in {source_images_dir}, skipping...")
                continue

            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            for image in selected_images:
                try:
                    shutil.copy2(os.path.join(source_images_dir, image), os.path.join(dest_images_dir, image))
                    label_name = os.path.splitext(image)[0] + ".txt"
                    source_label_path = os.path.join(source_labels_dir, label_name)
                    dest_label_path = os.path.join(dest_labels_dir, label_name)

                    if os.path.exists(source_label_path):
                        shutil.copy2(source_label_path, dest_label_path)
                except Exception as e:
                    logging.error(f"Error copying {image}: {e}")

        # Copy `data.yaml` if it exists
        source_yaml = os.path.join(source_batch, "data.yaml")
        dest_yaml = os.path.join(dest_batch, "data.yaml")
        if os.path.exists(source_yaml):
            try:
                shutil.copy2(source_yaml, dest_yaml)
            except Exception as e:
                logging.error(f"Error copying {source_yaml}: {e}")

    logging.info(f"Mini dataset created successfully at {dest_root}")

# ======================= PROCESS BATCHES =======================
def process_batches(batch_root):
    """Main function to process dataset batches."""
    logging.info(f"Starting batch processing in {batch_root}...")

    if not os.path.exists(batch_root):
        logging.error(f"Batch root directory {batch_root} does not exist.")
        return
    
    for batch in os.listdir(batch_root):
        batch_path = os.path.join(batch_root, batch)
        if os.path.isdir(batch_path):
            logging.info(f"Processing batch: {batch_path}")
            initialize_dataset_directory(batch_path)

    logging.info("Batch processing completed successfully.")
    print("Batch processing completed. Check logs/process.log for details.")

# ======================= MAIN EXECUTION =======================
if __name__ == "__main__":
    dataset_path = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\bdd100_dataset"
    mini_dataset_path = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\bdd100_mini"

    # Initialize dataset structure
    initialize_dataset_directory(dataset_path)

    # Process existing batches
    process_batches(dataset_path)

    # Create mini dataset from original
    create_mini_dataset(dataset_path, mini_dataset_path, num_images=10, num_batches=5)
