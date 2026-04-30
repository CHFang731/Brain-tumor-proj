#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import configure_runtime, ensure_dir, load_config
from brain_tumor.models.classification import create_classification_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auxiliary MRI classification inference.")
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument("--checkpoint", default="models/classification/best_classifier.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/classification_prediction.json")
    parser.add_argument("--device", default=None, help="Torch device string, e.g. cpu, cuda, cuda:1.")
    return parser.parse_args()


def main() -> None:
    configure_runtime(ROOT)
    import torch
    from PIL import Image
    from torchvision import transforms

    args = parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" in checkpoint:
        config = checkpoint["config"]

    image_size = int(config["training"]["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    requested_device = args.device or config.get("training", {}).get("device")
    if requested_device:
        device = torch.device(str(requested_device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_classification_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(image.to(device)), dim=1)[0].cpu()
    classes = list(config["classes"])
    result = {
        "image": args.image,
        "predicted_class": classes[int(probabilities.argmax().item())],
        "probabilities": {class_name: float(probabilities[idx]) for idx, class_name in enumerate(classes)},
    }
    output_path = ensure_dir(Path(args.output).parent) / Path(args.output).name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
