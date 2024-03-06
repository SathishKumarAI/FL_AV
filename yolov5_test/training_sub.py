# %%
import torch
import os
import subprocess

# # Change to a different directory
# os.chdir(r'C:\Users\SIU856522160\Downloads')
# # Get the new default path
# os.getcwd()

print(torch.__version__)
print(torch.cuda.is_available())
import gc
gc.collect()
torch.cuda.empty_cache()


# %%


completed_batches = []  # List to store completed batch numbers

def train_model(devices_folder, weights_path, epochs, img_size, device):
    global completed_batches  # Declare the global variable inside the function

    for device_folder in os.listdir(devices_folder):
        device_dir = os.path.join(devices_folder, device_folder)
        if not os.path.isdir(device_dir):
            continue

        train_dir = os.path.join(device_dir)
        batch_dirs = [batch for batch in os.listdir(train_dir) if batch.startswith('SubData_')]
        
        # Filter out completed batches for the current device
        batch_dirs = [batch for batch in batch_dirs if batch.split('_')[1] not in completed_batches]

        if len(batch_dirs) == 0:
            print(f"All batches for {device_folder} have already completed training. Skipping...")
            completed_batches.append(device_folder)
            continue

        selected_batch = batch_dirs[0]  # Select the first available batch for training
        
        batch_num = selected_batch.split('_')[1]
        dataset_yaml_path = f"dataset_{device_folder}_batch_{batch_num}.yaml"
        print(dataset_yaml_path, os.path.join(r"C:\Users\SIU856522160\Downloads\major\output_yaml\\",dataset_yaml_path))
        
    # import subprocess

    command = [
        "python", r"C:\Users\SIU856522160\Downloads\yolov5\train.py",
        "--batch", "8",
        "--data", os.path.join(r"C:\Users\SIU856522160\Downloads\major\output_yaml\\", dataset_yaml_path),
        "--weights", weights_path,
        "--epochs", str(epochs),
        "--img-size", str(img_size),
        "--device", str(device)
    ]

    # Execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the output during execution
    for line in iter(process.stdout.readline, b''):
        print(line.decode().strip())

    # Wait for the process to finish and capture the output
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Training completed for {device_folder}, Batch {batch_num}")
        completed_batches.append(f"{batch_num}")
    else:
        print(f"Error occurred during training for {device_folder}, Batch {batch_num}")
        print(stderr.decode())


    print("Completed batches:", completed_batches)


# %%
class_names = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]
output_dir = r"C:\Users\SIU856522160\Downloads\major\output_yaml"
devices_folder = r"C:\Users\SIU856522160\Downloads\yolov5\data\vech"
output_yaml_path = r"C:\Users\SIU856522160\Downloads\major\output_yaml"

weights_path = r"C:\Users\SIU856522160\Downloads\yolov5\yolov5s.pt"
epochs = 12
img_size = 1024
device = 0
# C:\Users\SIU856522160\Downloads\yolov5\train.py
train_model(devices_folder, weights_path, epochs, img_size, device)





