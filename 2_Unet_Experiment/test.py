# test_predictions.py

import os
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn

from SegmentationDataset import SegmentationDataset
from model import UNet
from centralized import load_data, evaluate  # Reusing functions from centralized.py


def test_random_predictions(num_samples=5, threshold=0.5):
    # Load test data using the centralized load_data function.
    # (We ignore the train_loader by using _)
    _, test_loader = load_data(batch_size=4, train_split=0.8)

    # Load the UNet model and put it on the correct device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Optionally load trained weights if available.
    model_path = "./models/global_FL_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found; using random-initialized weights.")

    model.eval()

    # Collect all samples from the test loader.
    test_samples = []
    for images, targets in test_loader:
        for i in range(images.size(0)):
            test_samples.append((images[i], targets[i]))

    if len(test_samples) < num_samples:
        num_samples = len(test_samples)
        print(f"Only {num_samples} samples available in the test set.")

    # Randomly select num_samples samples.
    selected_samples = random.sample(test_samples, num_samples)

    # Prepare the figure with num_samples rows and 3 columns: image, ground truth, prediction.
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # If there's only one sample, ensure axs is 2D.
    if num_samples == 1:
        axs = [axs]

    for idx, (img, target) in enumerate(selected_samples):
        img_batch = img.unsqueeze(0).to(device)  # Add batch dimension.
        with torch.no_grad():
            output = model(img_batch)
            pred = torch.sigmoid(output)
            pred = (pred > threshold).float()

        # Convert to NumPy arrays.
        img_np = img.cpu().squeeze().numpy()
        target_np = target.cpu().squeeze().numpy()
        pred_np = pred.cpu().squeeze().numpy()

        axs[idx][0].imshow(img_np, cmap="gray")
        axs[idx][0].set_title("Image")
        axs[idx][1].imshow(target_np, cmap="gray")
        axs[idx][1].set_title("Ground Truth")
        axs[idx][2].imshow(pred_np, cmap="gray")
        axs[idx][2].set_title("Prediction")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(15)  # Display for 5 seconds without blocking.
    plt.close(fig)
    print(f"Displayed {num_samples} random predictions.")


def compute_dice_stats(threshold=0.5):
    # Load test data.
    _, test_loader = load_data(batch_size=4, train_split=0.8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Load trained model weights if available.
    model_path = "./models/global_FL_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found; using random-initialized weights.")

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss, avg_dice, pixel_acc = evaluate(model, test_loader, criterion, device, threshold)
    print(f"Test Loss: {test_loss:.4f}, Average Dice: {avg_dice:.4f}, Pixel Accuracy: {pixel_acc:.2%}")


if __name__ == "__main__":

    print("Running random predictions test...")
    test_random_predictions(num_samples=5)

    print("\nComputing overall test set DICE statistics...")
    compute_dice_stats(threshold=0.5)
