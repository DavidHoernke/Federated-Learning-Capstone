# test_model.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from client import SimpleCNN  # Import the model architecture from client.py


def test_trained_model(model_path: str = "federated_model.pth"):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = SimpleCNN().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,  # Use test data
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test the model
    correct = 0
    total = 0

    print("\nTesting model on MNIST test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Results:")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    try:
        test_trained_model()
    except FileNotFoundError:
        print("Error: Could not find the model file 'federated_model.pth'")
        print("Make sure to run the server and complete training first.")