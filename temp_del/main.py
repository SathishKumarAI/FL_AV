import argparse
import subprocess
import time
import logging
import socket
import os
from typing import List
import threading

# Port configuration
PORTS = {
    "superlink": 9092,
    "server": 8080,
    "client_base": 9094,
    "deployment": 9093
}

def configure_logging():
    logging.basicConfig(
        filename='simulation.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_in_terminal(title: str, command: List[str], log_file: str = None):
    """Start a command in new terminal window with logging and conda env activation"""
    activate_cmd = 'conda activate flower2; '
    cmd = ['cmd', '/c', 'start', 'powershell', '-NoExit', '-Command']
    
    if log_file:
        ps_command = (
            f'$host.UI.RawUI.WindowTitle = "{title}"; '
            f'{activate_cmd} python {" ".join(command)} | Tee-Object -FilePath "{log_file}"'
        )
    else:
        ps_command = f'$host.UI.RawUI.WindowTitle = "{title}"; {activate_cmd} python {" ".join(command)}'
    
    cmd.append(ps_command)
    
    subprocess.Popen(
        cmd, 
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def tail_logs(log_files: List[str]):
    """Tail log files in the main terminal (Windows compatible)"""
    def tail(file: str):
        try:
            with open(file, 'r') as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    print(f"[{os.path.basename(file)}] {line.strip()}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    threads = []
    for log_file in log_files:
        if os.path.exists(log_file):
            t = threading.Thread(target=tail, args=(log_file,), daemon=True)
            t.start()
            threads.append(t)
        else:
            print(f"Log file {log_file} not found yet...")
    return threads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, required=True)
    args = parser.parse_args()

    configure_logging()
    logging.info("Starting federated learning system")

    # Generate list of log files to monitor
    log_files = ['server.log', 'simulation.log']
    for cid in range(args.num_clients):
        log_files.append(f'client_{cid}.log')

    try:
        # Start SuperLink
        start_in_terminal(
            "SuperLink",
            ["-m", "flwr.server.superlink", "--insecure", f"--port={PORTS['superlink']}"],
            "superlink.log"
        )
        time.sleep(2)

        # Start Server
        start_in_terminal(
            "Flower Server",
        ["server.py", 
            f"--server_address=0.0.0.0:{PORTS['server']}",
            f"--rounds=10",
            f"--min_clients={args.num_clients}"],
        "server.log"
        )
        time.sleep(5)

        # Start SuperNodes
        for cid in range(args.num_clients):
            client_port = PORTS['client_base'] + cid
            start_in_terminal(
            f"SuperNode-{cid}",
            ["-m", "flwr.server.supernode",
                "--insecure",
                f"--superlink=127.0.0.1:{PORTS['superlink']}",
                f"--node-config=partition-id={cid}",
                f"--client-app-address=127.0.0.1:{client_port}"],
            f"supernode_{cid}.log"
            )
            time.sleep(1)

        # Start Clients
        for cid in range(args.num_clients):
            start_in_terminal(
            f"Client-{cid}",
            ["FlowerClient.py",
                f"--cid={cid}",
                f"--server_address=127.0.0.1:{PORTS['server']}",
                f"--data_path=data/client_{cid}/data.yaml"],
            f"client_{cid}.log"
        )
            time.sleep(1)

        # Start monitoring
        print("Monitoring logs...")
        print("Press Ctrl+C to stop monitoring (components will keep running)")
        tail_threads = tail_logs(log_files)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️  Components are still running in separate windows!")
        print("⚠️  Close them manually or use:")
        print(f"⚠️  taskkill /F /IM python.exe /T")

if __name__ == "__main__":
    main()