import os
import requests
from zipfile import ZipFile

# Specify the target directory
target_directory = "/home/siu856522160/major/test/object_detection/model/yolov5/data/datasets"

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
# Example usage: bash data/scripts/get_coco128.sh
# parent
# ├── yolov5
# └── major
#     └── test
#         └── object_detection
#             └── model
#                 └── yolov5
#                     └── data
#                         └── datasets  ← downloads here

# Download/unzip images and labels
url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
download_path = os.path.join(target_directory, 'coco128.zip')

print(f'Downloading {url} ...')
response = requests.get(url, stream=True)
print("Download Completed.")
with open(download_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file.write(chunk)

with ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall(target_directory)

os.remove(download_path)  # Remove the downloaded zip file

# Print the entire path and list of files in the target directory
print(f"\nFiles in '{target_directory}' after download and extraction:")
file_list = os.listdir(target_directory)
for file_name in file_list:
    file_path = os.path.join(target_directory, file_name)
    print(file_path)
