"""Predict the occupancy of one pre-cropped CNRPark parking-space image."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import SmallParkingCNN, choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a single parking-space crop")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/final_model.pt"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SmallParkingCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    size = int(checkpoint["image_size"])
    image = Image.open(args.image).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    class_names = ["busy", "free"]
    predicted = int(np.argmax(probabilities))
    print(f"Prediction: {class_names[predicted]}")
    print(f"Confidence: {probabilities[predicted]:.4f}")
    print(f"busy={probabilities[0]:.4f}, free={probabilities[1]:.4f}")


if __name__ == "__main__":
    main()
