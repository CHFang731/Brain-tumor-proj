#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import load_config
from brain_tumor_seg.data import make_or_load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or load train/val/test split for segmentation.")
    parser.add_argument("--config", default="configs/segmentation_2d.yaml")
    parser.add_argument("--force", action="store_true", help="Regenerate split JSON even if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    data_cfg = config["data"]
    split_cfg = config["split"]

    split_json = ROOT / data_cfg["split_json"]
    if args.force and split_json.exists():
        split_json.unlink()

    split = make_or_load_split(
        dataset_root=ROOT / data_cfg["dataset_root"],
        split_json=split_json,
        train_fraction=float(split_cfg.get("train_fraction", 0.7)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        seed=int(config["training"].get("seed", 42)),
    )
    counts = split.get("meta", {}).get("counts", {})
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
