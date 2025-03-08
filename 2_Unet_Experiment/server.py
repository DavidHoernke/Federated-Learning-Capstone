import argparse
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from centralized import UNet
from StartConfig import *

# Initialize the global model
global_model = UNet(in_channels=1, out_channels=1)

# Custom FedAvg strategy with global model saving, evaluation, and early stopping.
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_dice = 0.0
        self.no_improvement_rounds = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list,
    ):
        """Aggregate model weights using weighted average and store checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated model...")
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(global_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            global_model.load_state_dict(state_dict, strict=True)
            model_path = f"global_model_round_{server_round}.pth"
            torch.save(global_model.state_dict(), model_path)
            print(f"Model checkpoint saved: {model_path}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: list,
    ):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        if aggregated_metrics is not None and "DICE Score" in aggregated_metrics:
            current_dice = aggregated_metrics["DICE Score"]
            print(f"Round {server_round} evaluation metrics: {aggregated_metrics}")
            if current_dice > self.best_dice:
                self.best_dice = current_dice
                self.no_improvement_rounds = 0
            else:
                self.no_improvement_rounds += 1
        else:
            print(f"Round {server_round}: No DICE Score available in aggregated metrics.")
        return aggregated_loss, aggregated_metrics

    def should_stop(self, server_round: int) -> bool: #Early stopping logic
        if self.no_improvement_rounds >= NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP and NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP!=0:
            print(f"Early stopping: Global DICE did not improve for {NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP} consecutive rounds.")
            return True
        return False


if __name__ == "__main__":

    # Create a strategy that waits for the specified number of clients.
    strategy = SaveModelStrategy(
        fraction_fit= CLIENT_PARTICIPATION_FRACTION,
    )

    # Start the Flower server; training will stop early if early stopping is triggered.
    fl.server.start_server(
        server_address="127.0.0.1:8000",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("Federated Learning Complete. Saving final model...")
    torch.save(global_model.state_dict(), "global_FL_model.pth")
    print("Final global model saved to global_FL_model.pth")
