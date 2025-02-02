import os
import yaml
import logging

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "process.log")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
    print("Logging initialized. Check logs/process.log for details.")
    logging.info("Logging setup complete.")

setup_logging()

# Optimized category mapping with correct indices (0-12)
CATEGORY_MAPPING = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "train": 5, "motorcycle": 6, "bicycle": 7, "traffic light": 8, 
    "traffic sign": 9, "trailer": 10, "other person": 11, "other vehicle": 12
}

# Ensure the 'names' list follows correct index order
CATEGORY_NAMES = [name for name, index in sorted(CATEGORY_MAPPING.items(), key=lambda item: item[1])]

def ensure_data_yaml_exists(batch_root):
    """Ensures that a valid data.yaml file exists in each batch directory and always overwrites it."""
    logging.info("Creating/updating data.yaml for all batches...")

    if not os.path.exists(batch_root):
        logging.error(f"Batch root directory {batch_root} does not exist.")
        return

    for batch in os.listdir(batch_root):
        batch_path = os.path.join(batch_root, batch)
        data_yaml_path = os.path.join(batch_path, 'data.yaml')

        if os.path.isdir(batch_path):
            try:
                logging.info(f"Overwriting data.yaml in {batch_path}...")
                data_yaml_content = {
                    'path': os.path.abspath(batch_path),
                    'train': os.path.join(batch_path, 'train.txt'),
                    'val': os.path.join(batch_path, 'val.txt'),
                    'test': os.path.join(batch_path, 'test.txt'),
                    'names': CATEGORY_NAMES  # Now stored as a properly formatted list
                }
                with open(data_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
                logging.info(f"data.yaml updated successfully in {batch_path}.")
            except Exception as e:
                logging.error(f"Error updating data.yaml in {batch_path}: {e}")

def create_splits(batch_root):
    """Creates or updates train.txt, val.txt, and test.txt files listing image paths."""
    logging.info("Creating/updating dataset split files (train.txt, val.txt, test.txt) for all batches...")

    if not os.path.exists(batch_root):
        logging.error(f"Batch root directory {batch_root} does not exist.")
        return

    for batch in os.listdir(batch_root):
        batch_path = os.path.join(batch_root, batch)
        if os.path.isdir(batch_path):
            train_file = os.path.join(batch_path, "train.txt")
            val_file = os.path.join(batch_path, "val.txt")
            test_file = os.path.join(batch_path, "test.txt")

            train_list, val_list, test_list = [], [], []

            for split in ["train", "val", "test"]:
                images_dir = os.path.join(batch_path, split, "images")
                if os.path.exists(images_dir):
                    image_files = [os.path.join(batch_path, split, "images", f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
                    if split == "train":
                        train_list.extend(image_files)
                    elif split == "val":
                        val_list.extend(image_files)
                    elif split == "test":
                        test_list.extend(image_files)

            try:
                with open(train_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(train_list) + '\n')
                logging.info(f"train.txt updated in {batch_path}.")
                
                with open(val_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(val_list) + '\n')
                logging.info(f"val.txt updated in {batch_path}.")
                
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(test_list) + '\n')
                logging.info(f"test.txt updated in {batch_path}.")

            except Exception as e:
                logging.error(f"Error updating dataset split files for {batch}: {e}")

def save_directory_structure(batch_root):
    """Scans each batch and saves the directory structure to a file."""
    logging.info("Saving directory structure for each batch...")

    if not os.path.exists(batch_root):
        logging.error(f"Batch root directory {batch_root} does not exist.")
        return

    for batch in os.listdir(batch_root):
        batch_path = os.path.join(batch_root, batch)
        structure_file = os.path.join(batch_path, "directory_structure.txt")

        if os.path.isdir(batch_path):
            try:
                logging.info(f"Writing directory structure in {batch_path}...")
                
                # Define expected files and their mapped names
                expected_files = {
                    "train": "train.txt",
                    "val": "val.txt",
                    "test": "test.txt"
                }

                structure_content = []
                for key, filename in expected_files.items():
                    file_path = os.path.join(batch_path, filename)
                    if os.path.exists(file_path):
                        structure_content.append(f"{key}: {filename}")

                # Write the structure to a file
                with open(structure_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(structure_content) + '\n')

                logging.info(f"directory_structure.txt saved in {batch_path}.")
            except Exception as e:
                logging.error(f"Error writing directory structure in {batch_path}: {e}")

def process_batches(batch_root):
    """Main function to process dataset batches."""
    logging.info("Starting batch processing... Overwriting all files with latest changes.")
    ensure_data_yaml_exists(batch_root)
    create_splits(batch_root)
    save_directory_structure(batch_root)
    logging.info("Batch processing completed successfully.")
    print("Batch processing completed. Check logs/process.log for details.")

if __name__ == "__main__":
    batch_root = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\bdd100_batch"  # Update this path if necessary
    process_batches(batch_root)
