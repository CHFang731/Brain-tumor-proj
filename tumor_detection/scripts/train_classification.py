#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import configure_runtime, ensure_dir, load_config
from brain_tumor.data.classification import discover_classification_items
from brain_tumor.models.classification import create_classification_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train auxiliary Kaggle MRI classification model.")
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cpu, cuda, cuda:1. Overrides training.device in config.",
    )
    return parser.parse_args()


class ImageFolderListDataset:
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from PIL import Image

        item = self.items[index]
        image = Image.open(item.image).convert("RGB")
        return self.transform(image), item.label


def main() -> None:
    configure_runtime(ROOT)
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import transforms
    from tqdm import tqdm

    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    train_config = config["training"]
    classes = list(config["classes"])
    output_dir = ensure_dir(data_config["output_dir"])

    torch.manual_seed(int(train_config.get("seed", 42)))
    image_size = int(train_config["image_size"])
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_items = discover_classification_items(data_config["dataset_root"], "Training", classes)
    test_items = discover_classification_items(data_config["dataset_root"], "Testing", classes)
    train_loader = DataLoader(
        ImageFolderListDataset(train_items, train_tfms),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 4)),
    )
    test_loader = DataLoader(
        ImageFolderListDataset(test_items, eval_tfms),
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 4)),
    )

    requested_device = args.device or train_config.get("device")
    if requested_device:
        device = torch.device(str(requested_device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA index {device.index}, but only {torch.cuda.device_count()} device(s) are visible."
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        resolved_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Using device: cuda:{resolved_index} ({torch.cuda.get_device_name(resolved_index)})")
    else:
        print("Using device: cpu")

    model = create_classification_model(config).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_config.get("amp", True)) and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(ROOT / "runs" / "classification"))
    best_accuracy = -1.0
    best_epoch = -1
    best_path = output_dir / "best_classifier.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"train epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())

        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"eval epoch {epoch}"):
                logits = model(images.to(device))
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        accuracy = sum(int(a == b) for a, b in zip(all_preds, all_labels)) / max(1, len(all_labels))
        train_loss = total_loss / max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_accuracy": accuracy,
            "best_accuracy": max(best_accuracy, accuracy),
        }
        history.append(record)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/test", accuracy, epoch)
        print(record)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "accuracy": accuracy}, best_path)
            report = {
                "accuracy": accuracy,
                "best_epoch": epoch,
                "classification_report": classification_report(all_labels, all_preds, target_names=classes, output_dict=True),
                "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
            }
            with (output_dir / "classification_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    writer.close()
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
