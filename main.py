# main.py
import subprocess

# Start the server
server_proc = subprocess.Popen(["python", "server.py"])

# Start the clients
client_procs = [
    subprocess.Popen(["python", "client.py"]),
    subprocess.Popen(["python", "client.py"]),
]

# Wait for the server process to terminate
server_proc.wait()

# Terminate client processes
for proc in client_procs:
    proc.terminate()
    proc.wait()
