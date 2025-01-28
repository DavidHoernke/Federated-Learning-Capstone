import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input: 1x28x28, Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Input: 32x28x28, Output: 64x28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                 # Output: 64x14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)                           # Fully connected layer
        self.fc2 = nn.Linear(128, 10)                                     # 10 classes for MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy

# Federated client class
class MyFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        return self.get_parameters(config=None), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, self.criterion, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

# Main function
if __name__ == "__main__":
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")

    # Transformations and dataset loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model initialization
    model = SimpleCNN().to(device)

    # Create and start the Flower client
    client = MyFlowerClient(model, train_loader, test_loader, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)