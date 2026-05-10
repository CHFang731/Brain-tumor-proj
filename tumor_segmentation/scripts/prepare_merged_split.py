#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a train-expanded split by appending distribution-matched LGG MRI slices."
    )
    parser.add_argument("--base-split", default="data/processed/split_2d.json")
    parser.add_argument(
        "--lgg-root",
        default="../kagglehub_cache/datasets/mateuszbuda/lgg-mri-segmentation/versions/2/kaggle_3m",
    )
    parser.add_argument("--output", default="data/processed/split_2d_plus_lgg600_matched.json")
    parser.add_argument("--seed", type=int, default=56)
    parser.add_argument("--max-additional", type=int, default=600)
    parser.add_argument("--max-per-subject", type=int, default=8)
    parser.add_argument("--mean-min", type=float, default=0.08)
    parser.add_argument("--mean-max", type=float, default=0.24)
    parser.add_argument("--std-min", type=float, default=0.10)
    parser.add_argument("--std-max", type=float, default=0.22)
    parser.add_argument("--mask-area-min", type=float, default=0.001)
    parser.add_argument("--mask-area-max", type=float, default=0.08)
    return parser.parse_args()


def load_stats(image_path: Path, mask_path: Path) -> tuple[float, float, float]:
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    mask = (np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) > 127).astype(np.float32)
    return float(image.mean()), float(image.std()), float(mask.mean())


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    base_split_path = (root / args.base_split).resolve()
    lgg_root = (root / args.lgg_root).resolve()
    output_path = (root / args.output).resolve()

    with base_split_path.open("r", encoding="utf-8") as handle:
        base = json.load(handle)

    rng = random.Random(args.seed)
    candidates: list[tuple[Path, Path, str, float, float, float]] = []
    for mask_path in sorted(lgg_root.rglob("*_mask.tif")):
        image_path = mask_path.with_name(mask_path.name.replace("_mask.tif", ".tif"))
        if not image_path.exists():
            continue
        subject = mask_path.parent.name
        mean, std, area = load_stats(image_path, mask_path)
        if not (args.mean_min <= mean <= args.mean_max):
            continue
        if not (args.std_min <= std <= args.std_max):
            continue
        if not (args.mask_area_min <= area <= args.mask_area_max):
            continue
        candidates.append((image_path, mask_path, subject, mean, std, area))

    rng.shuffle(candidates)
    per_subject = defaultdict(int)
    selected: list[tuple[Path, Path, str, float, float, float]] = []
    for row in candidates:
        if len(selected) >= args.max_additional:
            break
        subject = row[2]
        if per_subject[subject] >= args.max_per_subject:
            continue
        per_subject[subject] += 1
        selected.append(row)

    extra_items: list[dict[str, str]] = []
    for image_path, mask_path, subject, _mean, _std, _area in selected:
        stem = image_path.stem
        extra_items.append(
            {
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "id": f"lgg_{subject}_{stem}",
            }
        )

    merged_train = list(base["train"]) + extra_items
    out = {
        "train": merged_train,
        "val": base["val"],
        "test": base["test"],
        "meta": {
            **base.get("meta", {}),
            "base_split": str(base_split_path),
            "added_dataset": str(lgg_root),
            "added_count": len(extra_items),
            "max_additional": args.max_additional,
            "max_per_subject": args.max_per_subject,
            "seed": args.seed,
            "filter": {
                "mean": [args.mean_min, args.mean_max],
                "std": [args.std_min, args.std_max],
                "mask_area": [args.mask_area_min, args.mask_area_max],
            },
            "counts": {
                "train": len(merged_train),
                "val": len(base["val"]),
                "test": len(base["test"]),
                "train_base": len(base["train"]),
                "train_added": len(extra_items),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "train_base": len(base["train"]),
                "train_added": len(extra_items),
                "train_total": len(merged_train),
                "val_total": len(base["val"]),
                "test_total": len(base["test"]),
                "candidates_before_cap": len(candidates),
                "subjects_used": len(per_subject),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
