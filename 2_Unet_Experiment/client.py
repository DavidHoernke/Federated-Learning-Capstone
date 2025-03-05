import sys  # To pass client ID and number of clients from command line
from collections import OrderedDict

import flwr as fl
import torch

from model import UNet
from centralized import load_data, evaluate, run_training

# Get client ID from command-line argument (first argument)
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# Get number of clients from command-line argument (second argument), default to 50 if not provided
num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 50

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # Detect and set the device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(in_channels=1, out_channels=1).to(self.device)  # Move model to device
        # Use both client_id and num_clients in data loading
        self.train_loader, self.test_loader = load_data(client_id=client_id, num_clients=num_clients)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        run_training(
            model=self.net,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            num_epochs=1,
            device=self.device
        )
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, accuracy, pixelAcc = evaluate(self.net, self.test_loader, criterion, device=self.device)
        return test_loss, len(self.test_loader.dataset), {"DICE Score": accuracy, "Pixel Accuracy": pixelAcc}

# Start the Flower client, connecting to the server at 127.0.0.1:8000
fl.client.start_numpy_client(
    server_address="127.0.0.1:8000",
    client=FlowerClient()
)
