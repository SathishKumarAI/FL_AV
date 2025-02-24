# 🚀 Federated Object Detection with YOLOv8 and Flower  

**Collaboratively train YOLOv8 models on distributed datasets while preserving data privacy.**  

---

## 🌐 Table of Contents  
- [🔒 Why Federated Learning?](#-why-federated-learning)  
- [🤖 Why YOLOv8?](#-why-yolov8)  
- [⚙️ Tech Stack](#️-tech-stack)  
- [📥 Installation](#-installation)  
  - [1. Prerequisites](#1-prerequisites)  
  - [2. Set Up Conda Environment](#2-set-up-conda-environment)  
  - [3. Install Dependencies](#3-install-dependencies)  
- [📂 Dataset Preparation](#-dataset-preparation)  
  - [1. Download Preprocessed Data](#1-download-preprocessed-data)  
  - [2. Partition Data for Federated Clients](#2-partition-data-for-federated-clients)  
- [🚀 Quick Start](#-quick-start)  
  - [1. Launch the Flower Server](#1-launch-the-flower-server)  
  - [2. Start Federated Clients](#2-start-federated-clients)  
  - [3. Monitor Training](#3-monitor-training)  
- [🎛️ Simulation Setup](#️-simulation-setup)  
  - [1. Set Up a Flower Simulation Project](#1-set-up-a-flower-simulation-project)  
  - [4. Clean Up](#4-clean-up)  
- [🏗️ Project Architecture](#️-project-architecture)  
- [🛠️ Troubleshooting](#️-troubleshooting)  
- [📚 References](#-references)  
- [🗺️ Future Roadmap](#️-future-roadmap)  

---

## 🔒 Why Federated Learning?  
- **Data Privacy**: Sensitive data (e.g., surveillance footage, vehicle sensors) stays on-device.  
- **Bandwidth Efficiency**: Only model gradients (not raw images) are transmitted.  
- **Regulatory Compliance**: Ideal for GDPR, HIPAA, or industry-specific data policies.  
- **Edge Optimization**: Train models directly on edge devices (cameras, drones, IoT sensors).  

---

## 🤖 Why YOLOv8?  
- **State-of-the-Art Performance**: Outperforms YOLOv5 in accuracy and speed.  
- **Multi-Task Support**: Object detection, segmentation, and classification.  
- **Scalability**: Pre-trained models (`yolov8n`, `yolov8s`, etc.) for diverse hardware.  
- **Ease of Use**: Simplified training API and extensive documentation.  

---

## ⚙️ Tech Stack  
- **Frameworks**: [Flower](https://flower.dev) (FL), [Ultralytics YOLOv8](https://ultralytics.com)  
- **Dataset**: [BDD100K](https://bdd-data.berkeley.edu/) (preprocessed and hosted on Google Drive)  
- **GPU Support**: CUDA 11.x, NVIDIA Drivers  
- **Tools**: Conda, Git, WSL2 (optional)  

---

## 📥 Installation  

### 1. Prerequisites  
- **Python 3.10+**  
- **NVIDIA GPU** with CUDA 11.8+  
- **Git** and **Conda**  

### 2. Set Up Conda Environment  
```bash  
conda create -n fl_yolov8 python=3.10 -y  
conda activate fl_yolov8  
```  

### 3. Install Dependencies  
```bash  
# PyTorch with CUDA  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  

# YOLOv8 and Flower  
pip install ultralytics flwr[simulation]  

# Additional utilities  
pip install opencv-python numpy tqdm pyyaml  
```  

---

## 📂 Dataset Preparation  

### 1. Download Preprocessed Data  
The BDD100K dataset (already in YOLOv8 format) is hosted on Google Drive:  
🔗 [Download Dataset](https://drive.google.com/drive/folders/1R-lelZR3LBgeHfMlRR_OhOIzfUuxPBcZ?usp=sharing)  

```bash  
mkdir -p federated_yolov8/data  
mv ~/Downloads/bdd100k_yolov8.zip federated_yolov8/data/  
cd federated_yolov8/data  
unzip bdd100k_yolov8.zip  
```  

### 2. Partition Data for Federated Clients  
Split the dataset into client-specific subsets using `split_clients.py`:  
```bash  
python split_clients.py \  
    --source="data" \  
    --output="data_clients" \  
    --num_clients=10  
```  

Each client directory requires a `data.yaml` file. Example for `client_0`:  
```yaml  
train: ../client_0/train/images  
val: ../client_0/val/images  
nc: 13  # Number of classes  
names: ["car", "person", "bus", "traffic light", ...]  
```  

---

## 🚀 Quick Start  

### 1. Launch the Flower Server  
```bash  
python server.py --num_rounds=10 --batch_size=32  
```  

### 2. Start Federated Clients  
Open separate terminals for each client:  
```bash  
# Client 0  
python client.py --id=0 --data_path="data_clients/client_0/data.yaml"  

# Client 1  
python client.py --id=1 --data_path="data_clients/client_1/data.yaml"  
```  

### 3. Monitor Training  
- **Server Logs**: Global model accuracy, round duration, client participation.  
- **Client Logs**: Local training loss, validation mAP, GPU utilization.  

---

## 🎛️ Simulation Setup  

Flower provides a **simulation engine** to test federated learning on a single machine.

### 1. Set Up a Flower Simulation Project  
```bash  
flwr new my-project --framework PyTorch --username flower  
cd my-project  
pip install -e .  
flwr run .  
```  

### 4. Clean Up  
Use **Ctrl+C** in each terminal to stop the processes.

---

## 🏗️ Project Architecture  

| File | Purpose |  
|------|---------|  
| `server.py` | Flower server for aggregating client updates. |  
| `client.py` | Flower client to train YOLOv8 locally. |  
| `simulation.py` | Runs multiple simulated FL clients on a single machine. |  
| `train_config.py` | Configures training hyperparameters. |  
| `split_clients.py` | Partitions dataset into client-specific subsets. |  
| `label_utils.py` | Converts BDD100K annotations to YOLO format. |  

---

## 🛠️ Troubleshooting  

| Issue | Solution |  
|-------|----------|  
| **CUDA Out of Memory** | Reduce `BATCH_SIZE` or use `yolov8n`. |  
| **No GPU Detected** | Verify `torch.cuda.is_available()` and reinstall PyTorch with CUDA. |  
| **Dataset Path Errors** | Ensure `data.yaml` paths match the client directory structure. |  
| **Dependency Conflicts** | Use a fresh Conda environment. |  

---

## 📚 References  
- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com)  
- **Flower**: [Official Documentation](https://flower.dev/docs)  
- **BDD100K**: [Dataset Paper](https://arxiv.org/abs/1805.04687)  

---

## 🗺️ Future Roadmap  
1. **Advanced FL Strategies**: Implement FedProx/FedNova for non-IID data.  
2. **Edge Deployment**: Optimize for NVIDIA Jetson/Raspberry Pi.  
3. **Real-Time Inference**: On-device inference with periodic FL updates.  
4. **Multi-Task Learning**: Add segmentation support with YOLOv8.  

---

Thanks! 😊
