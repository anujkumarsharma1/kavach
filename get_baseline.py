"""
Phase 1 — Load a real pretrained CNN and save a clean baseline.
Run this first: python get_baseline.py
"""
import glob
import json

import torch
import torchvision
from PIL import Image


def load_sample_images(folder="sample_images"):
    paths = sorted(
        glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.jpeg") + glob.glob(f"{folder}/*.png")
    )
    if len(paths) < 3:
        raise RuntimeError(
            f"Found {len(paths)} images in {folder}/. Drop at least 3 of your own photos "
            f"in there before running this script."
        )
    return paths


def predict(model, weights_meta, image_paths):
    transforms = weights_meta.transforms()
    categories = weights_meta.meta["categories"]
    results = {}
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        batch = transforms(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(batch)[0], dim=0)
        top_idx = int(torch.argmax(probs))
        results[path] = {"class": categories[top_idx], "confidence": float(probs[top_idx])}
    return results


if __name__ == "__main__":
    weights_meta = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights_meta)
    model.eval()

    torch.save(model.state_dict(), "clean_model.pt")
    print("Saved clean_model.pt (44.7MB, ImageNet-pretrained ResNet18).")

    images = load_sample_images()
    baseline = predict(model, weights_meta, images)

    with open("baseline_predictions.json", "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline predictions on {len(images)} images:")
    for path, result in baseline.items():
        print(f"  {path}: {result['class']} ({result['confidence']:.1%})")