import flwr as fl
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
import threading
import time
import sys


class ManualControlStrategy(FedAvg):
    def __init__(
            self,
            *args,
            min_clients: int = 2,
            num_rounds: int = 3,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.training_completed = threading.Event()
        self.current_round = 0
        self.final_parameters: Optional[Parameters] = None
        self.min_clients = min_clients
        self.start_training = threading.Event()
        self.connected_clients = 0
        self.num_rounds = num_rounds

    def initialize_parameters(self, client_manager):
        """Wait for minimum number of clients before starting."""
        print(f"\rWaiting for clients to connect... (0/{self.min_clients})", end="")
        sys.stdout.flush()

        while True:
            current_clients = len(client_manager.clients)
            if current_clients != self.connected_clients:
                self.connected_clients = current_clients
                print(f"\rWaiting for clients to connect... ({self.connected_clients}/{self.min_clients})", end="")
                sys.stdout.flush()

                if self.connected_clients >= self.min_clients:
                    print("\n\nMinimum number of clients connected!")
                    input("Press Enter to start training...")
                    self.start_training.set()
                    break
            time.sleep(0.1)

        return super().initialize_parameters(client_manager)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager
    ) -> List[tuple[ClientProxy, Dict]]:
        """Wait for training start signal before first round."""
        if server_round == 1:
            self.start_training.wait()
            print("\nStarting training...\n")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate model updates from clients."""
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.current_round = server_round
        print(f"\nCompleted round {server_round}/{self.num_rounds}")

        if aggregated and server_round == self.num_rounds:
            try:
                parameters = aggregated[0]
                self.final_parameters = parameters
                save_model(parameters, "federated_model.pth")
                print("\nTraining completed! Final model saved to federated_model.pth")
                self.training_completed.set()
            except Exception as e:
                print(f"Error saving final model: {str(e)}")
                import traceback
                print(traceback.format_exc())

        return aggregated


def save_model(parameters, model_path: str):
    """Save the model parameters to a file."""
    try:
        if parameters:
            # Extract the tensors from the parameters
            if isinstance(parameters, list):
                # If parameters is already a list of tensors/arrays
                tensors = parameters
            else:
                # If parameters is a Flower Parameters object
                tensors = parameters.tensors

            # Convert parameters to PyTorch tensors
            processed_params = []
            for tensor in tensors:
                if isinstance(tensor, bytes):
                    # If tensor is in bytes format, convert to numpy array
                    param_array = np.frombuffer(tensor, dtype=np.float32)
                    processed_params.append(param_array)
                else:
                    # If tensor is already a numpy array
                    param_array = np.asarray(tensor, dtype=np.float32)
                    processed_params.append(param_array)

            # Create the state dictionary
            params_dict = OrderedDict({
                f"param_{k}": torch.tensor(v)
                for k, v in enumerate(processed_params)
            })

            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

            # Save the model
            torch.save(params_dict, model_path)
            print(f"Successfully saved model to {model_path}")

    except Exception as e:
        print(f"Error saving model to {model_path}: {str(e)}")
        print("\nDebug Info:")
        print(f"Type of parameters: {type(parameters)}")
        if hasattr(parameters, 'tensors'):
            print(f"Number of tensors: {len(parameters.tensors)}")
        import traceback
        print(traceback.format_exc())


def start_server(num_rounds: int = 3, min_clients: int = 2):
    strategy = ManualControlStrategy(
        evaluate_fn=None,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        min_clients=min_clients,
        num_rounds=num_rounds
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    return strategy


if __name__ == "__main__":
    MIN_CLIENTS = 2
    NUM_ROUNDS = 3

    print("Starting Flower server...")
    print(f"Minimum clients required: {MIN_CLIENTS}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print("\nWaiting for clients to connect...")

    try:
        strategy = start_server(num_rounds=NUM_ROUNDS, min_clients=MIN_CLIENTS)
    except KeyboardInterrupt:
        print("\nServer stopped by user")