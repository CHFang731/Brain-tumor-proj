#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average compatible PyTorch segmentation checkpoints.")
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths relative to repo root.")
    parser.add_argument("--weights", default="", help="Optional comma-separated averaging weights.")
    parser.add_argument("--output", required=True, help="Output checkpoint path relative to repo root.")
    parser.add_argument("--config", required=True, help="Config path to store in checkpoint metadata.")
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    checkpoint_paths = [ROOT / item for item in parse_csv(args.checkpoints)]
    if args.weights:
        weights = [float(item) for item in parse_csv(args.weights)]
    else:
        weights = [1.0] * len(checkpoint_paths)
    if len(weights) != len(checkpoint_paths):
        raise ValueError("--weights count must match --checkpoints count")
    total = sum(weights)
    weights = [weight / total for weight in weights]

    soups: dict[str, torch.Tensor] = {}
    metadata = []
    for ckpt_path, weight in zip(checkpoint_paths, weights, strict=True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"]
        metadata.append(
            {
                "checkpoint": str(ckpt_path),
                "weight": float(weight),
                "best_val_dice": float(ckpt.get("best_val_dice", ckpt.get("val_metrics", {}).get("dice", -1.0))),
            }
        )
        for name, tensor in state.items():
            if not torch.is_floating_point(tensor):
                if name not in soups:
                    soups[name] = tensor.clone()
                continue
            part = tensor.float() * float(weight)
            soups[name] = part if name not in soups else soups[name] + part

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": soups,
            "config": args.config,
            "soup": metadata,
            "best_val_dice": max(item["best_val_dice"] for item in metadata),
        },
        output_path,
    )
    print(json.dumps({"output": str(output_path), "members": metadata}, ensure_ascii=True))


if __name__ == "__main__":
    main()
