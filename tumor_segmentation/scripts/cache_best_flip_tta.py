#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split
from brain_tumor_seg.model import build_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str
    checkpoint: str
    image_size: int
    in_channels: int


BEST_SPECS = [
    ModelSpec("old", "configs/segmentation_2d_smp_long.yaml", "models/smp_unet_resnet34/best_unet2d.pt", 256, 1),
    ModelSpec("s44", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed44.yaml", "models/smp_deeplabv3plus_resnet50_320_seed44/best_unet2d.pt", 320, 3),
    ModelSpec("s45", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed45.yaml", "models/smp_deeplabv3plus_resnet50_320_seed45/best_unet2d.pt", 320, 3),
    ModelSpec("effb4ft", "configs/segmentation_2d_smp_deeplabv3plus_effb4_320_from256_ft.yaml", "models/smp_deeplabv3plus_effb4_320_from256_ft/best_unet2d.pt", 320, 3),
    ModelSpec("s384", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed50_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed50_ft/best_unet2d.pt", 384, 3),
    ModelSpec("s384s45", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed45_conservative_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed45_conservative_ft/best_unet2d.pt", 384, 3),
    ModelSpec("s384s56", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed56_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed56_ft/best_unet2d.pt", 384, 3),
    ModelSpec("s384s64", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed64_from_seed45c_polish_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed64_from_seed45c_polish_ft/best_unet2d.pt", 384, 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache flip-TTA probability maps for the current best ensemble specs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--names", default="", help="Optional comma-separated subset of spec names.")
    return parser.parse_args()


def resize_to(prob: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if prob.shape[-2:] == size:
        return prob
    return F.interpolate(prob, size=size, mode="bilinear", align_corners=False)


def flip_tta_prob(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    p0 = torch.sigmoid(model(x))
    p1 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[3])), dims=[3]))
    p2 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[2])), dims=[2]))
    p3 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[2, 3])), dims=[2, 3]))
    return (p0 + p1 + p2 + p3) / 4.0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ref_hw = (int(args.ref_size), int(args.ref_size))
    cache_dir = ensure_dir(ROOT / args.cache_dir)
    selected = {name.strip() for name in args.names.split(",") if name.strip()}
    specs = [spec for spec in BEST_SPECS if not selected or spec.name in selected]

    split_cfg = load_config(ROOT / "configs/segmentation_2d_smp_long.yaml")
    split = make_or_load_split(
        dataset_root=ROOT / split_cfg["data"]["dataset_root"],
        split_json=ROOT / split_cfg["data"]["split_json"],
        train_fraction=float(split_cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(split_cfg["split"].get("val_fraction", 0.15)),
        seed=int(split_cfg["training"].get("seed", 42)),
    )

    for spec in specs:
        cfg = load_config(ROOT / spec.config)
        model = build_model(cfg).to(device)
        state = torch.load(ROOT / spec.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        for split_name in ("val", "test"):
            out_path = cache_dir / f"{spec.name}_{split_name}_ref{args.ref_size}.pt"
            if out_path.exists():
                print(f"exists {spec.name} {split_name}: {out_path}", flush=True)
                continue
            ds = BrainTumorSegDataset(
                split[split_name],
                image_size=spec.image_size,
                training=False,
                in_channels=spec.in_channels,
            )
            loader = DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(cfg["data"].get("num_workers", 4)),
                pin_memory=True,
            )
            chunks = []
            with torch.no_grad():
                for batch in loader:
                    x = batch["image"].to(device, non_blocking=True)
                    chunks.append(resize_to(flip_tta_prob(model, x).cpu(), ref_hw).half())
            probs = torch.cat(chunks, dim=0)
            torch.save(probs, out_path)
            print(f"saved {spec.name} {split_name}: {tuple(probs.shape)}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
