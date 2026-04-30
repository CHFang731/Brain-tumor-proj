#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split
from brain_tumor_seg.metrics import binary_metrics
from brain_tumor_seg.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on val or test split.")
    parser.add_argument("--config", default="configs/segmentation_2d.yaml")
    parser.add_argument("--checkpoint", default="models/unet2d/best_unet2d.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    config = load_config(ROOT / args.config)
    data_cfg = config["data"]

    split = make_or_load_split(
        dataset_root=ROOT / data_cfg["dataset_root"],
        split_json=ROOT / data_cfg["split_json"],
        train_fraction=float(config["split"].get("train_fraction", 0.7)),
        val_fraction=float(config["split"].get("val_fraction", 0.15)),
        seed=int(config["training"].get("seed", 42)),
    )

    ds = BrainTumorSegDataset(
        split[args.split],
        image_size=int(data_cfg.get("image_size", 256)),
        training=False,
        in_channels=int(config.get("model", {}).get("in_channels", 1)),
        stronger_aug=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    device = resolve_device(args.device)
    model = build_model(config).to(device)

    ckpt = torch.load(ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    agg = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            m = binary_metrics(logits, masks)
            for k in agg:
                agg[k] += m[k]

    n = max(1, len(loader))
    result = {k: v / n for k, v in agg.items()}
    result.update(
        {
            "split": args.split,
            "num_samples": len(ds),
            "checkpoint": str((ROOT / args.checkpoint).resolve()),
            "device": str(device),
        }
    )

    report_dir = ensure_dir(ROOT / "reports")
    out_path = report_dir / f"metrics_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
