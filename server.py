# server.py
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
        self.current_round = server_round
        print(f"\nCompleted round {server_round}/{self.num_rounds}")
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_after_training(self, server_round: int, results, failures):
        aggregated = super().aggregate_after_training(server_round, results, failures)
        if aggregated:
            self.final_parameters = aggregated
            self.training_completed.set()
            # Save the model after training completes
            save_model(aggregated, "federated_model.pth")
            print("\nTraining completed! Model saved to federated_model.pth")
        return aggregated

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

def save_model(parameters, model_path: str = "federated_model.pth"):
    """Save the final model parameters to a file."""
    if parameters:
        params_dict = zip(range(len(parameters)), parameters)
        state_dict = OrderedDict({f"param_{k}": torch.tensor(v) for k, v in params_dict})
        torch.save(state_dict, model_path)
        print(f"Model saved to {model_path}")
    else:
        print("No parameters available to save")

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