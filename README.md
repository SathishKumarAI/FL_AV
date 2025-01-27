
# **Project**
Why Federated Learning over a Centralized Model?

Federated Learning (FL) allows each data owner (or device) to keep its sensitive data locally, sending only model parameters (gradients) to a central server. This approach not only enhances privacy by avoiding the need for a massive centralized data repository, but also lowers bandwidth costs and reduces the risk of data leaks. In many real-world applications-such as autonomous vehicles or edge-based surveillance-transferring large volumes of raw images to a single server can be impractical or raise confidentiality concerns. FL addresses these challenges by keeping the training data closer to its source.

Why Object Detection?

Object detection is a fundamental task in computer vision, enabling systems to recognize and localize objects within images or video frames. By federating an object detection model, we allow multiple independent data silos---such as cameras from different traffic intersections or vehicle fleets-to collaboratively train a model without ever pooling their raw images. This is critical in scenarios like smart city infrastructure, where privacy laws may restrict data sharing between agencies, or fleet management, where each vehicle has limited connectivity and must remain operationally secure. FL with object detection meets both practical deployment needs (e.g., edge devices with limited bandwidth) and regulatory requirements (data privacy, security) while still benefiting from diverse, real-world data for improved model robustness.

## **Overview**

This repository demonstrates how to perform **federated learning** for **object detection** with [Flower](https://flower.dev), **YOLOv5** (via [Ultralytics](https://github.com/ultralytics/ultralytics)), and a **BDD100K** dataset split. The setup uses **Python**, **Conda**, **CUDA**, and **Git Bash** on **Windows** (with optional **WSL2** support).

**Key Highlights:**

1. **Federated Approach**  
   - Multiple clients (nodes) each hold a subset of the data (BDD100K images/labels).  
   - Each client trains locally (YOLOv5) and sends model updates to a central server (Flower).  
   - The server aggregates updates (e.g., via FedAvg) and broadcasts new global weights back to clients.

2. **BDD100K Dataset**  
   - **Largest driving dataset** with 100K videos and a variety of tasks (including object detection).  
   - Highly diverse (geography, weather, environment), making models more robust.  
   - References:  
       - [Official BDD100K Website](https://bdd-data.berkeley.edu/)  
       - [About the Dataset](https://doc.bdd100k.com/download.html)  
       - [Papers with Code link](https://paperswithcode.com/dataset/bdd100k)  
       - [Original Paper (ArXiv)](https://arxiv.org/abs/1805.04687)

3. **YOLOv5 Model**  
   - Provided by [Ultralytics](https://github.com/ultralytics/ultralytics).  
   - Used for high-performance object detection tasks.  
   - Pretrained checkpoints like `yolov5s.pt`, `yolov5m.pt`, etc.

4. **Project Files**  
   - `train_config.py` – Global config (num rounds, batch size, etc.)  
   - `label_utils.py` – BDD100K → YOLO label conversion & verification  
   - `data_utils.py` – Helpers for data paths and `data.yaml` verification  
   - `client.py` – Federated client code (runs local YOLO training)  
   - `server.py` – Federated server code (coordinates aggregation with Flower)  
   - `run_federation.bat` – Windows batch script to launch the server and multiple clients  
   - `requirements.txt` – Python dependencies

---

## **Resources**

Below are the primary resources you will need:

1. **Dataset**:  
   - [BDD100K](https://bdd-data.berkeley.edu/)  
   - Download the images and any JSON/COCO annotations you need for your object detection tasks.  

2. **Dependencies/Installations** (see next section for detailed steps):  
   - **Python 3.10.8.6** (or a similar 3.10+ release)  
   - **Git Bash**  
   - **Conda** (Miniconda/Anaconda)  
   - **CUDA Toolkit** (matching your GPU drivers)  
   - **WSL2** (optional, but can help if you prefer a Linux environment on Windows)

---

## **Installations**

Below is the recommended approach to ensure a clean environment and avoid module conflicts. Check out the provided documentation links for more detailed instructions.

1. **Install Python 3.10.8.6**  
   - [Download for Windows](https://www.python.org/downloads/windows/)  
   - Make sure to enable “Add Python to PATH” during installation (or handle manually).

2. **Git Bash Setup**  
   - Refer to [`installations/docs/Git_setup.md`](installations/docs/Git_setup.md).  
   - Git Bash offers a handy terminal environment on Windows.

3. **Conda Environment**  
   - See [`installations/docs/GitBash_Conda_Setup.md`](installations/docs/GitBash_Conda_Setup.md).  
   - Example commands to create and activate a new Conda environment:
     ```bash
     conda create -n federated_yolov5 python=3.10 -y
     conda activate federated_yolov5
     ```
   - We recommend using a **separate environment** so that your dependencies do not clash with other projects.

4. **CUDA Installations**  
   - Check out [`installations/docs/CUDA_installations.md`](installations/docs/CUDA_installations.md).  
   - You need an NVIDIA GPU and matching drivers (e.g., RTX 4070) for efficient YOLO training.

5. **WSL-2 (Optional)**  
   - Detailed guide in [`installations/docs/WSL.md`](installations/docs/WSL.md).  
   - WSL2 allows you to run a Linux-like environment inside Windows, which can simplify some tool installations.

---

## **Repository Setup**

### **1. Git Clone**

Make sure you have configured Git with SSH or HTTPS. Refer to [this StackOverflow link](https://stackoverflow.com/questions/68775869/message-support-for-password-authentication-was-removed) if you encounter issues:

```bash
git clone git@github.com:yourusername/federated_yolov5.git
cd federated_yolov5
```

*(Replace the repo URL with your actual GitHub repository.)*

### **2. Install Python Dependencies**

From inside the cloned repo, install the required packages:

```bash
conda activate federated_yolov5

# Option 1: Use pip directly
pip install -r requirements.txt

# Option 2: Or install them one by one, especially if you need custom PyTorch wheels
# pip install flwr[simulation]>=1.8.0
# pip install torch>=2.2.1 torchvision>=0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
# pip install ultralytics>=8.2.0 opencv-python-headless numpy tqdm pyyaml
```

> **Note**: Make sure the version of PyTorch you install matches your CUDA version (e.g., CUDA 11.8 for an RTX 4070).

---

## **Prepare Data**

1. **Download or Acquire BDD100K**  
   - Official site: [BDD100K](https://bdd-data.berkeley.edu/)  
   - Once downloaded, you should have images (possibly 100k) and annotation files.  

2. **Convert Labels (If Needed)**  
   - If your BDD100K annotations are in JSON format, convert them to YOLO format using `label_utils.py`.  
   - Example:
     ```python
     from label_utils import convert_bdd_to_yolo

     # Suppose you have a file "bdd_annotations.json" and want to convert it
     convert_bdd_to_yolo(
         bdd_annotation_file="path/to/bdd_annotations.json",
         output_label_dir="data/labels/",
         class_names=["car", "bus", "person", ...]
     )
     ```
   - This step ensures each image has a corresponding `.txt` file in the YOLO bounding-box specification (i.e., `<class_index> <x_center> <y_center> <width> <height>` in [0,1]).

3. **Partition Data for Federated Clients**  
   - Create subfolders like `client_0`, `client_1`, …, `client_9`.  
   - Inside each client folder, create `train/images`, `train/labels`, `val/images`, and `val/labels`.  
   - Move or copy your subset of images/labels into these subfolders.  
   - Provide a `data.yaml` for each client that points to those folders:
     ```yaml
     # data/client_0/data.yaml (example)
     train: data/client_0/train/images
     val: data/client_0/val/images
     nc: 10
     names: ["car","person","traffic_light","bus", ...]
     ```

> **Note**: More detailed instructions on your data-splitting approach are in [update.md](update.md) (placeholder). You can adapt or rename that file as needed.

---

## **Project Architecture**

### **Files & Directories**

1. **`server.py`**  
   - Flower server orchestrating federated training and aggregation.  

2. **`client.py`**  
   - Flower client that trains a local YOLOv5 model using each client’s portion of data.  

3. **`train_config.py`**  
   - Central place for hyperparameters: number of rounds, local epochs, batch size, etc.  

4. **`label_utils.py`**  
   - Functions to convert BDD100K bounding boxes into YOLO format.  
   - Verifies correctness of YOLO label files.  

5. **`data_utils.py`**  
   - Helpers for locating and verifying `data.yaml` for each client.  

6. **`run_federation.bat`**  
   - Windows batch script launching one server process and multiple client processes in separate CMD windows.  

7. **`requirements.txt`**  
   - Lists pinned or recommended packages.  

---

## **Step-by-Step Usage**

1. **Activate Environment**  
   ```bash
   conda activate federated_yolov5
   ```

2. **(Optional) Label Conversion**  
   ```bash
   python -c "import label_utils; label_utils.convert_bdd_to_yolo('bdd_annotations.json', 'data/labels', ['car','bus','person'])"
   ```
   - Adjust the annotation file path and classes as needed.

3. **Organize Partitions**  
   - Create `data/client_0`, `data/client_1`, …, each with `train` and `val` subfolders and a `data.yaml`.  

4. **Run Federated Training**  
   - **Windows**: Double-click or run:
     ```bash
     ./run_federation.bat
     ```
   - This will:  
     1. Launch `server.py` in a new window.  
     2. Wait a few seconds.  
     3. Launch 10 client windows (`client.py --id=0` through `client.py --id=9`).  

5. **Monitor Logs**  
   - Each client window displays YOLO training logs.  
   - The server window shows progress on each federated round.

6. **Result**  
   - After **NUM_ROUNDS** (in `train_config.py`), the global model parameters converge.  
   - You can see each round’s aggregated performance (mAP, etc.) in the logs.

---

## **Computational Setup**

- **Operating System**: Windows 10/11  
- **GPU**: e.g., NVIDIA RTX 4070 or similar (ensure CUDA drivers match)  
- **Python**: 3.10.x  
- **Memory Requirements**:  
  - 10 concurrent clients with YOLOv5 training may require substantial GPU VRAM.  
  - If OOM errors occur, lower the `BATCH_SIZE` or run fewer clients concurrently.

---

## **Model Used**

We utilize **YOLOv5** from [Ultralytics](https://github.com/ultralytics/ultralytics) (e.g., `yolov5s.pt` base weights), which provides:

- High-speed inference.  
- Good performance on real-world object detection tasks.  
- Straightforward fine-tuning & customization.

---

## **References**

1. **BDD100K**  
   - [Dataset Homepage](https://bdd-data.berkeley.edu/)  
   - [Documentation & Download](https://doc.bdd100k.com/download.html)  
   - [Papers with Code](https://paperswithcode.com/dataset/bdd100k)  
   - [Paper (ArXiv)](https://arxiv.org/abs/1805.04687)  

2. **YOLOv5**  
   - [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)  

3. **Flower**  
   - [Official Site](https://flower.dev)  
   - [GitHub Repo](https://github.com/adap/flower)  

4. **Environment Setup Docs**  
   - [`installations/docs/Git_setup.md`](installations/docs/Git_setup.md)  
   - [`installations/docs/GitBash_Conda_Setup.md`](installations/docs/GitBash_Conda_Setup.md)  
   - [`installations/docs/CUDA_installations.md`](installations/docs/CUDA_installations.md)  
   - [`installations/docs/WSL.md`](installations/docs/WSL.md)

---

## **Future Plans**

- **Extend** to YOLOv8 or custom detection architectures.  
- **Incorporate** advanced federated algorithms (e.g., FedProx, FedNova).  
- **Use** more robust data splits or real-time streaming of images.  
- **Experiment** with half-precision vs. full-precision trade-offs.

---

### **End of Document**
