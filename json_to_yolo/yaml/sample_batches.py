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
            print(f"Skipping batch {batch_num}, source directory missing: {source_batch}")
            continue

        os.makedirs(dest_batch, exist_ok=True)

        for split in ["train", "val", "test"]:
            source_split = os.path.join(source_batch, split, "images")
            dest_split_images = os.path.join(dest_batch, split, "images")
            dest_split_labels = os.path.join(dest_batch, split, "labels")

            os.makedirs(dest_split_images, exist_ok=True)
            os.makedirs(dest_split_labels, exist_ok=True)

            if not os.path.exists(source_split):
                print(f"Skipping {split} in batch {batch_num}, source split missing: {source_split}")
                continue

            # Get all image files
            image_files = [f for f in os.listdir(source_split) if f.endswith(('.jpg', '.png'))]
            if not image_files:
                print(f"No images found in {source_split}")
                continue

            # Randomly select `num_images` images
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            for image in selected_images:
                source_image_path = os.path.join(source_split, image)
                dest_image_path = os.path.join(dest_split_images, image)

                # Copy the image
                shutil.copy2(source_image_path, dest_image_path)

                # Find and copy the corresponding label
                label_name = os.path.splitext(image)[0] + ".txt"
                source_label_path = os.path.join(source_batch, split, "labels", label_name)
                dest_label_path = os.path.join(dest_split_labels, label_name)

                if os.path.exists(source_label_path):
                    shutil.copy2(source_label_path, dest_label_path)

        # Copy data.yaml
        source_yaml = os.path.join(source_batch, "data.yaml")
        dest_yaml = os.path.join(dest_batch, "data.yaml")
        if os.path.exists(source_yaml):
            shutil.copy2(source_yaml, dest_yaml)

    print(f"Mini dataset created successfully at {dest_root}")

# Define source and destination paths
source_root = r"C:\Users\sathish\Downloads\FL_AV\my-project\data\bdd100_batch"
dest_root = r"C:\Users\sathish\Downloads\FL_AV\my-project\data\bdd100_mini"

# Run the function to create the mini dataset
create_mini_dataset(source_root, dest_root, num_images=10, num_batches=5)
