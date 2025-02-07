import os
import shutil
import random

def create_mini_dataset(source_root, dest_root, num_images=10, num_batches=3):
    """Creates a mini version of the dataset by randomly selecting images and copying their labels."""
    
    if not os.path.exists(source_root):
        print(f"Error: Source directory '{source_root}' does not exist.")
        return

    os.makedirs(dest_root, exist_ok=True)

    for batch_num in range(1, num_batches + 1):
        source_batch = os.path.join(source_root, f"batch_{batch_num}")
        dest_batch = os.path.join(dest_root, f"batch_{batch_num}")

        if not os.path.exists(source_batch):
            print(f"Skipping batch {batch_num}, missing: {source_batch}")
            continue

        os.makedirs(dest_batch, exist_ok=True)

        for split in ["train", "val", "test"]:
            source_images_dir = os.path.join(source_batch, split, "images")
            source_labels_dir = os.path.join(source_batch, split, "labels")
            dest_images_dir = os.path.join(dest_batch, split, "images")
            dest_labels_dir = os.path.join(dest_batch, split, "labels")

            # Skip if images directory is missing
            if not os.path.exists(source_images_dir):
                print(f"Skipping {split} in batch {batch_num}, missing: {source_images_dir}")
                continue

            os.makedirs(dest_images_dir, exist_ok=True)
            os.makedirs(dest_labels_dir, exist_ok=True)

            # List image files
            image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png'))]
            if not image_files:
                print(f"No images found in {source_images_dir}, skipping...")
                continue

            # Select up to `num_images` randomly
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            for image in selected_images:
                try:
                    # Copy image
                    shutil.copy2(os.path.join(source_images_dir, image), os.path.join(dest_images_dir, image))

                    # Copy corresponding label if it exists
                    label_name = os.path.splitext(image)[0] + ".txt"
                    source_label_path = os.path.join(source_labels_dir, label_name)
                    dest_label_path = os.path.join(dest_labels_dir, label_name)

                    if os.path.exists(source_label_path):
                        shutil.copy2(source_label_path, dest_label_path)
                except Exception as e:
                    print(f"Error copying {image}: {e}")

        # Copy `data.yaml` if it exists
        source_yaml = os.path.join(source_batch, "data.yaml")
        dest_yaml = os.path.join(dest_batch, "data.yaml")
        if os.path.exists(source_yaml):
            try:
                shutil.copy2(source_yaml, dest_yaml)
            except Exception as e:
                print(f"Error copying {source_yaml}: {e}")

    print(f"Mini dataset created successfully at {dest_root}")

# Define source and destination paths
source_root = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\bdd100_batch"
dest_root = r"C:\Users\sathish\Downloads\FL_ModelForAV\my-project\data\bdd100_mini9"

# Run the function to create the mini dataset
create_mini_dataset(source_root, dest_root, num_images=10, num_batches=5)
