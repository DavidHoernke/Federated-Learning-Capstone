import sys  # To pass client ID from command line
from collections import OrderedDict

import flwr as fl
import torch

from model import UNet
from centralized import load_data, evaluate, run_training

# Get client ID from command-line argument
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
num_clients = 2  # Adjust based on how many clients you are running

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # Detect and set the device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet().to(self.device)  # Move model to device
        self.train_loader, self.test_loader = load_data(client_id=client_id, num_clients=num_clients)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        # Pass the device parameter explicitly to run_training
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
        # Use self.device here as well
        test_loss, accuracy, pixelAcc = evaluate(self.net, self.test_loader, criterion, device=self.device)
        # Return a tuple with 3 elements: (loss, number of examples, metrics dict)
        return test_loss, len(self.test_loader.dataset), {"DICE Score": accuracy, "Pixel Accuracy": pixelAcc}

# Start client with unique dataset
fl.client.start_numpy_client(
    server_address="127.0.0.1:8000",
    client=FlowerClient()
)
