# Setting Up GPU Environment for PyTorch with CUDA

## Download CUDA Toolkit and cuDNN

* Download the [Cuda toolkit for version 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

* Download cuDNN v8.5.0 (August 8th, 2022), for [CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)

## PyTorch Installation

To install PyTorch with GPU support:

* Run the following command in the Anaconda Prompt:

    ```bash
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ```

    [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

* Alternatively, use the following pip command for Conda Torch installations:

    ```bash
    pip install torch torchvision torchaudio
    ```

### Additional Information:

- Ensure that the downloaded CUDA Toolkit and cuDNN versions are compatible with PyTorch.

- Adjust the CUDA version in the installation commands based on your specific requirements.

- Check the PyTorch documentation and CUDA compatibility matrix for any updates or additional information.

