import json
import cv2
import torch
import logging
from torch.utils.data import Dataset
import os
from model.yolov5.utils.augmentations import augment_hsv, letterbox


def load_data():
  """
  Loads images and labels from the client's data folder.

  Retrieves the client ID from the environment variable 'CLIENT_ID', logs
  the number and names of read files, and returns images and labels.

  Returns:
      tuple: A tuple of (images, labels) where:
          images: A list of PyTorch tensors representing the images.
          labels: A list of lists holding bounding box coordinates and class IDs.
  """

  # Get client ID from environment variable
  try:
    client_id = int(os.environ.get("CLIENT_ID"))
  except (ValueError, TypeError):
    raise RuntimeError("Invalid CLIENT_ID environment variable")

  # Define your data folder structure based on the partitioning scheme
  data_folder = f"client_{client_id}"  # Replace with your actual structure
  image_folder = os.path.join(data_folder, "images")
  label_folder = os.path.join(data_folder, "labels")

  # Initialize empty lists for images and labels
  images = []
  labels = []

  # Log the number of images and labels folders
  logging.info(f"Found {len(os.listdir(image_folder))} images in {image_folder}")
  logging.info(f"Found {len(os.listdir(label_folder))} labels in {label_folder}")

  # Iterate through image and label files
  for image_filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_filename)
    label_path = os.path.join(label_folder, image_filename[:-4] + ".txt")

    # Load image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Load label from text file
    with open(label_path, "r") as f:
      label_data = f.readlines()

    # Parse label data and convert to bounding boxes and class IDs
    for line in label_data:
      class_id, x1, y1, x2, y2 = map(int, line.strip().split())
      label = [x1, y1, x2, y2, class_id]  # Convert to YOLOv5 format
      labels.append(label)

    # Log the filenames
    logging.debug(f"Read image: {image_filename}")
    logging.debug(f"Read label: {label_path[:-4] + '.txt'}")

    # Apply YOLOv5 transformations (replace with your chosen YOLOv5 methods)
    image, label = augment_hsv(image, label)  # Example YOLOv5 augmentation
    image, label = letterbox(image, label, 640, auto=True, scaleup=True)

    # Convert image to PyTorch tensor and normalize
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)  # Convert to CHW format

    # Add image and label to the lists
    images.append(image)
    labels.append(label)

  return images, labels
