# Starter Script Overview

The `starter.py` script is your single command to launch both the server and client processes for your federated learning experiment. It passes the number of clients to both the server and each client, and opens each client in its own terminal window.

## How It Works

- **Starter Script (`starter.py`):**
  - **Usage:**  
    Run from your project folder in a terminal:
    ```bash
    python starter.py 50
    ```
    This starts the server with 50 clients and spawns 50 client processes (client IDs 0 to 49).
  
  - **Server Launch:**  
    The script calls:
    ```bash
    python server.py --num_clients 50
    ```
    This passes the total number of clients as a command-line argument to the server. In `server.py`, the Flower server uses this value (via `argparse`) to wait until all clients are connected before starting training rounds.

  - **Client Launch:**  
    The script spawns each client in a new terminal window using Windows’ `start` command:
    ```bash
    start cmd /k python "client.py" <client_id> 50
    ```
    Each client receives its unique ID and the total number of clients. In `client.py`, these arguments are used to load the appropriate data partition and configure the client.

## Where to Run

- **Run From:**  
  Open your terminal (Command Prompt on Windows) and navigate to the directory containing `starter.py`, `server.py`, and `client.py`. Then run the command:
  ```bash
  python starter.py 50
