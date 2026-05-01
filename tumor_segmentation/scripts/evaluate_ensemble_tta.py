#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split
from brain_tumor_seg.metrics import binary_metrics
from brain_tumor_seg.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2-model ensemble with TTA on val/test split.")
    parser.add_argument("--config-a", default="configs/segmentation_2d_smp_long.yaml")
    parser.add_argument("--checkpoint-a", default="models/smp_unet_resnet34/best_unet2d.pt")
    parser.add_argument("--in-channels-a", type=int, default=1)
    parser.add_argument("--image-size-a", type=int, default=256)

    parser.add_argument("--config-b", default="configs/segmentation_2d_smp_unet_resnet34_256_focal.yaml")
    parser.add_argument("--checkpoint-b", default="models/smp_unet_resnet34_256_focal/best_unet2d.pt")
    parser.add_argument("--in-channels-b", type=int, default=3)
    parser.add_argument("--image-size-b", type=int, default=256)

    parser.add_argument("--weight-a", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="reports/ensemble_tta_metrics.json")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tta_prob(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    p0 = torch.sigmoid(model(x))
    p1 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[3])), dims=[3]))
    p2 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[2])), dims=[2]))
    p3 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[2, 3])), dims=[2, 3]))
    return (p0 + p1 + p2 + p3) / 4.0


def resize_prob(prob: torch.Tensor, ref_hw: tuple[int, int]) -> torch.Tensor:
    if prob.shape[-2:] == ref_hw:
        return prob
    return torch.nn.functional.interpolate(prob, size=ref_hw, mode="bilinear", align_corners=False)


def main() -> None:
    args = parse_args()

    cfg_a = load_config(ROOT / args.config_a)
    cfg_b = load_config(ROOT / args.config_b)

    split = make_or_load_split(
        dataset_root=ROOT / cfg_b["data"]["dataset_root"],
        split_json=ROOT / cfg_b["data"]["split_json"],
        train_fraction=float(cfg_b["split"].get("train_fraction", 0.7)),
        val_fraction=float(cfg_b["split"].get("val_fraction", 0.15)),
        seed=int(cfg_b["training"].get("seed", 42)),
    )
    items = split[args.split]

    ds_a = BrainTumorSegDataset(
        items,
        image_size=args.image_size_a,
        training=False,
        in_channels=args.in_channels_a,
    )
    ds_b = BrainTumorSegDataset(
        items,
        image_size=args.image_size_b,
        training=False,
        in_channels=args.in_channels_b,
    )
    loader_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = resolve_device(args.device)

    model_a = build_model(cfg_a).to(device)
    state_a = torch.load(ROOT / args.checkpoint_a, map_location=device)
    model_a.load_state_dict(state_a["model_state"])
    model_a.eval()

    model_b = build_model(cfg_b).to(device)
    state_b = torch.load(ROOT / args.checkpoint_b, map_location=device)
    model_b.load_state_dict(state_b["model_state"])
    model_b.eval()

    w_a = float(args.weight_a)
    w_b = 1.0 - w_a

    agg = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    with torch.no_grad():
        for batch_a, batch_b in zip(loader_a, loader_b):
            x_a = batch_a["image"].to(device, non_blocking=True)
            x_b = batch_b["image"].to(device, non_blocking=True)
            y = batch_a["mask"].to(device, non_blocking=True)

            p_a = tta_prob(model_a, x_a)
            p_b = tta_prob(model_b, x_b)
            target_hw = tuple(y.shape[-2:])
            p_a = resize_prob(p_a, target_hw)
            p_b = resize_prob(p_b, target_hw)
            p = w_a * p_a + w_b * p_b
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            logits = torch.log(p / (1 - p))

            m = binary_metrics(logits, y, threshold=float(args.threshold))
            for k in agg:
                agg[k] += m[k]

    n = max(1, len(loader_a))
    result = {k: v / n for k, v in agg.items()}
    result.update(
        {
            "split": args.split,
            "num_samples": len(items),
            "weight_a": w_a,
            "weight_b": w_b,
            "threshold": float(args.threshold),
            "config_a": args.config_a,
            "config_b": args.config_b,
            "checkpoint_a": str((ROOT / args.checkpoint_a).resolve()),
            "checkpoint_b": str((ROOT / args.checkpoint_b).resolve()),
            "device": str(device),
        }
    )

    out_path = ROOT / args.output
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
