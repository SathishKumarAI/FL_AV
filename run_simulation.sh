#!/usr/bin/env bash
WORKER_NUM=$1
# # Initialize and activate the Conda environment
# eval "$(conda shell.bash hook)"
# conda activate venv
# Calculate the process number by adding 1 to the worker number
PROCESS_NUM=$((WORKER_NUM + 1))
echo "Number of processes: $PROCESS_NUM"
# Write the hostname to a file named mpi_host_file
hostname > mpi_host_file
# Specify the full path to the Python interpreter within your Conda environment
PYTHON_PATH=$(which python3)
echo "Number of PYTHON_PATH: $PYTHON_PATH"
# Use the correct syntax for mpirun
mpirun -np $PROCESS_NUM $PYTHON_PATH main_fedml_object_detection.py --cf config/simulation/fedml_config.yaml
