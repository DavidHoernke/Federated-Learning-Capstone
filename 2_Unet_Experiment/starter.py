import subprocess
import sys
import os

def start_server(num_clients):
    server_path = os.path.abspath("server.py")
    # Pass the number of clients to the server via --num_clients argument
    cmd = f'python "{server_path}" --num_clients {num_clients}'
    proc = subprocess.Popen(cmd, shell=True)
    return proc

def start_clients(num_clients):
    processes = []
    client_path = os.path.abspath("client.py")
    for i in range(num_clients):
        # Pass the client id and total number of clients as arguments to client.py
        cmd = f'start cmd /k python "{client_path}" {i} {num_clients}'
        proc = subprocess.Popen(cmd, shell=True)
        processes.append(proc)
    return processes

if __name__ == "__main__":
    # Read the number of clients from the command line; default to 50 if not provided.
    if len(sys.argv) > 1:
        num_clients = int(sys.argv[1])
    else:
        num_clients = 2

    processes = []
    # Start the server process and add it to the list
    server_proc = start_server(num_clients)
    processes.append(server_proc)
    # Start all client processes and extend the process list
    client_procs = start_clients(num_clients)
    processes.extend(client_procs)

    # Wait for all processes to finish
    for proc in processes:
        proc.wait()
