#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download segmentation dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        default="nikhilroxtomar/brain-tumor-segmentation",
        help="Kaggle dataset handle",
    )
    parser.add_argument(
        "--cache-dir",
        default="../kagglehub_cache",
        help="Directory used as KAGGLEHUB_CACHE",
    )
    return parser.parse_args()


def main() -> None:
    import kagglehub

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    path = kagglehub.dataset_download(args.dataset)
    print(path)


if __name__ == "__main__":
    main()
