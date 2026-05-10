#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split
from brain_tumor_seg.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache flip-TTA probability maps for one model.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
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

    cfg = load_config(ROOT / args.config)
    split_cfg = load_config(ROOT / "configs/segmentation_2d_smp_long.yaml")
    split = make_or_load_split(
        dataset_root=ROOT / split_cfg["data"]["dataset_root"],
        split_json=ROOT / split_cfg["data"]["split_json"],
        train_fraction=float(split_cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(split_cfg["split"].get("val_fraction", 0.15)),
        seed=int(split_cfg["training"].get("seed", 42)),
    )
    model = build_model(cfg).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    image_size = int(cfg["data"].get("image_size", 256))
    in_channels = int(cfg.get("model", {}).get("in_channels", 1))
    for split_name in ("val", "test"):
        out_path = cache_dir / f"{args.name}_{split_name}_ref{args.ref_size}.pt"
        ds = BrainTumorSegDataset(split[split_name], image_size=image_size, training=False, in_channels=in_channels)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        chunks = []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                chunks.append(resize_to(flip_tta_prob(model, x).cpu(), ref_hw).half())
        probs = torch.cat(chunks, dim=0)
        torch.save(probs, out_path)
        print(f"saved {args.name} {split_name}: {tuple(probs.shape)}", flush=True)


if __name__ == "__main__":
    main()
