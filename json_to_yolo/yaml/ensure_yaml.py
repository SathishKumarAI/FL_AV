import os
import shutil
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

def ensure_data_yaml_exists(batch_path):
    logging.info("Creating/updating data.yaml...")
    data_yaml_path = os.path.join(batch_path, 'data.yaml')
    data_yaml_content = {
        'path': os.path.abspath(batch_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CATEGORY_NAMES),
        'names': CATEGORY_NAMES
    }
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
    logging.info(f"data.yaml updated successfully in {batch_path}.")

def create_splits(batch_path):
    logging.info("Creating dataset split files (train.txt, val.txt, test.txt) and copying images/labels...")
    splits = {"train": [], "val": [], "test": []}
    for split in splits.keys():
        images_dir = os.path.join(batch_path, "images", split)
        labels_dir = os.path.join(batch_path, "labels", split)
        target_images_dir = os.path.join(batch_path, "processed_images", split)
        target_labels_dir = os.path.join(batch_path, "processed_labels", split)
        
        if os.path.exists(target_images_dir):
            shutil.rmtree(target_images_dir)
        if os.path.exists(target_labels_dir):
            shutil.rmtree(target_labels_dir)
        
        os.makedirs(target_images_dir, exist_ok=True)
        os.makedirs(target_labels_dir, exist_ok=True)
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            for file in os.listdir(images_dir):
                if file.endswith(('.jpg', '.png')):
                    img_src_path = os.path.join(batch_path, "images", split, file)
                    label_src_path = os.path.join(batch_path, "labels", split, file.replace('.jpg', '.txt').replace('.png', '.txt'))
                    img_dest_path = os.path.join(target_images_dir, file)
                    label_dest_path = os.path.join(target_labels_dir, file.replace('.jpg', '.txt').replace('.png', '.txt'))
                    
                    if os.path.exists(label_src_path):
                        splits[split].append(img_src_path)
                        shutil.copy(img_src_path, img_dest_path)
                        shutil.copy(label_src_path, label_dest_path)
                    else:
                        logging.warning(f"Label missing for {file}, skipping...")
    
    for split, files in splits.items():
        with open(os.path.join(batch_path, f"{split}.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(files) + '\n')
        logging.info(f"{split}.txt updated in {batch_path}.")

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
    batch_root = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\"  # Update this path if necessary
    process_batches(batch_root)
