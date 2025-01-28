import flwr as fl

def start_server():
    # Define the server configuration
    server_config = fl.server.ServerConfig(num_rounds=3)

    # Start the Flower server
    # fl.server.start_server(server_address="127.0.0.1:8080", config=server_config)

    def evaluate(server_round, parameters, config):
        # Dummy evaluation logic, replace with your actual implementation
        loss = 0.5  # Replace with your evaluation loss
        accuracy = 0.8  # Replace with your evaluation accuracy
        return loss, {"accuracy": accuracy}

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=server_config,
        strategy=fl.server.strategy.FedAvg(evaluate_fn=evaluate)
    )

if __name__ == "__main__":
    start_server()
