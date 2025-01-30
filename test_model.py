import torch
from client import SimpleCNN


def test_saved_model(model_path):
    # Load the model architecture
    model = SimpleCNN()

    # Load the saved parameters
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    print("Model loaded successfully!")
    return model


# Test the saved model
model = test_saved_model("federated_model.pth")