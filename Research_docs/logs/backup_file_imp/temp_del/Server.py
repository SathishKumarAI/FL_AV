import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP runtime conflicts

import subprocess
import logging
import psutil  # For killing processes
import signal
import sys
import flwr as fl
from util import load_model, get_model_parameters, parameters_to_ndarrays

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

def kill_process_using_port(port):
    """Kill the process using the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logging.info(f"Killing process {proc.pid} using port {port}...")
                        proc.kill()
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logging.error(f"Error killing process using port {port}: {e}")

def start_server(port=8081):
    """Start the Flower server on the specified port."""
    try:
        # Kill any process using the port
        kill_process_using_port(port)

        # Load initial model
        model = load_model("cuda:0")  # Load YOLOv5 model
        initial_parameters = get_model_parameters(model.model)  # Get initial parameters

        # Define Flower strategy
        strategy = fl.server.strategy.FedAvg(
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda rnd: {"batch_size": 16, "epochs": 2},  # Pass config to clients
            evaluate_fn=evaluate_global_model,  # Server-side validation
            round_timeout=60,  # Timeout after 60 seconds
        )

        # Start the Flower server using the flower-superlink CLI
        logging.info(f"Starting Flower server on port {port}...")
        server_proc = subprocess.Popen([
            "flower-superlink",
            "--insecure",
            f"--port={port}"  # Use the specified port
        ])
        logging.info(f"Flower server started successfully on port {port}.")
        return server_proc
    except Exception as e:
        logging.error(f"Failed to start Flower server: {e}")
        raise

def evaluate_global_model(server_round, parameters, config):
    """Evaluate the global model on the server's validation set."""
    try:
        model = load_model("cuda:0")
        set_weights(model.model, parameters_to_ndarrays(parameters))
        results = model.val(data="config/val_data.yaml", imgsz=640, batch=16, device="cuda:0")
        metrics = {
            "mp": results.results_dict["metrics/precision"],
            "mr": results.results_dict["metrics/recall"],
            "map50": results.results_dict["metrics/mAP_0.5"],
            "map": results.results_dict["metrics/mAP_0.5:0.95"],
        }
        return results.results_dict["metrics/loss"], metrics
    except Exception as e:
        logging.error(f"Error during global model evaluation: {e}")
        return None, {}

def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    logging.info("Shutting down server gracefully...")
    sys.exit(0)

def main():
    """Start the Flower server."""
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    server_proc = None
    try:
        # Start the server on port 8081
        server_proc = start_server(port=8081)

        # Wait for the server process to finish
        server_proc.wait()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if server_proc:
            server_proc.terminate()

if __name__ == "__main__":
    main()