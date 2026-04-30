#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import configure_runtime, ensure_dir, load_config
from brain_tumor.data.brats import make_or_load_split
from brain_tumor.data.transforms import segmentation_transforms
from brain_tumor.models.segmentation import create_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation checkpoint.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--checkpoint", default="models/segmentation/best_segresnet.pt")
    parser.add_argument("--output", default="reports/segmentation_metrics.json")
    return parser.parse_args()


def region_dice(pred, target, region: str) -> float:
    import torch

    if region == "whole_tumor":
        pred_mask = pred > 0
        target_mask = target > 0
    elif region == "tumor_core":
        pred_mask = torch.logical_or(pred == 1, pred == 3)
        target_mask = torch.logical_or(target == 1, target == 3)
    elif region == "enhancing_tumor":
        pred_mask = pred == 3
        target_mask = target == 3
    else:
        raise ValueError(f"Unknown region: {region}")

    intersection = torch.logical_and(pred_mask, target_mask).sum().float()
    denominator = pred_mask.sum().float() + target_mask.sum().float()
    if denominator.item() == 0:
        return 1.0
    return float((2.0 * intersection / denominator).item())


def main() -> None:
    configure_runtime(ROOT)
    import torch
    from monai.data import DataLoader, Dataset
    from monai.inferers import sliding_window_inference
    from tqdm import tqdm

    args = parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" in checkpoint:
        config = checkpoint["config"]

    data_config = config["data"]
    train_config = config["training"]
    split = make_or_load_split(
        data_config["brats_root"],
        data_config["split_json"],
        seed=int(train_config.get("seed", 42)),
    )
    roi_size = tuple(int(v) for v in train_config["roi_size"])
    val_ds = Dataset(split["val"], transform=segmentation_transforms(roi_size, training=False))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(data_config.get("num_workers", 4)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_segmentation_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    regions = ("whole_tumor", "tumor_core", "enhancing_tumor")
    scores: dict[str, list[float]] = {region: [] for region in regions}
    subject_scores: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="evaluate"):
            image = batch["image"].to(device)
            label_onehot = batch["label"].to(device)
            logits = sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=int(train_config.get("sw_batch_size", 2)),
                predictor=model,
            )
            pred = logits.argmax(dim=1)[0].cpu()
            label = label_onehot.argmax(dim=1)[0].cpu()
            subject_id = batch.get("subject_id", ["unknown"])[0]
            record: dict[str, float | str] = {"subject_id": subject_id}
            for region in regions:
                score = region_dice(pred, label, region)
                scores[region].append(score)
                record[region] = score
            subject_scores.append(record)

    metrics = {
        "mean": {region: sum(values) / max(1, len(values)) for region, values in scores.items()},
        "subjects": subject_scores,
        "target": {"whole_tumor_dice": 0.90},
    }
    output_path = ensure_dir(Path(args.output).parent) / Path(args.output).name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics["mean"], indent=2))
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
