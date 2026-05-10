#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split


DEFAULT_MODELS = ("old", "s44", "s45", "effb4ft", "s384", "s384s45", "s384s56", "s384s64")
BEST_WEIGHTS = {
    "old": 0.12698717301282703,
    "s44": 0.1338425658410686,
    "s45": 0.04212298739392233,
    "effb4ft": 0.158983941016059,
    "s384": 0.2206164260063713,
    "s384s45": 0.158983941016059,
    "s384s56": 0.14836297581368268,
    "s384s64": 0.010099989900010101,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pixel-level logistic stacker from cached segmentation probabilities.")
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--d4-cache-dir", default="")
    parser.add_argument("--include-d4", action="store_true")
    parser.add_argument("--model-names", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--output", default="reports/cached_stacker_sweep_20260509.json")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--max-pos", type=int, default=600000)
    parser.add_argument("--max-hard-neg", type=int, default=600000)
    parser.add_argument("--max-random-neg", type=int, default=200000)
    parser.add_argument("--hard-neg-min-prob", type=float, default=0.02)
    parser.add_argument("--c", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--min-sizes", default="64,80,88,92,96,98,100,104,112,128")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_ground_truth(ref_size: int) -> dict[str, np.ndarray]:
    ref_cfg = load_config(ROOT / "configs/segmentation_2d_smp_long.yaml")
    split = make_or_load_split(
        dataset_root=ROOT / ref_cfg["data"]["dataset_root"],
        split_json=ROOT / ref_cfg["data"]["split_json"],
        train_fraction=float(ref_cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(ref_cfg["split"].get("val_fraction", 0.15)),
        seed=int(ref_cfg["training"].get("seed", 42)),
    )
    gt = {}
    for split_name in ("val", "test"):
        ds = BrainTumorSegDataset(split[split_name], image_size=ref_size, training=False, in_channels=1)
        gt[split_name] = torch.stack([ds[i]["mask"] for i in range(len(ds))], dim=0).numpy().astype(bool)
    return gt


def load_feature_cache(
    cache_dir: Path,
    d4_cache_dir: Path | None,
    model_names: list[str],
    ref_size: int,
    include_d4: bool,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    feature_names: list[str] = []
    probs: dict[str, dict[str, np.ndarray]] = {}

    def load_one(feature_name: str, path_prefix: Path, model_name: str) -> None:
        feature_names.append(feature_name)
        probs[feature_name] = {}
        for split_name in ("val", "test"):
            path = path_prefix / f"{model_name}_{split_name}_ref{ref_size}.pt"
            tensor = torch.load(path, map_location="cpu")
            probs[feature_name][split_name] = tensor.cpu().numpy().astype(np.float32, copy=False)
            print(f"loaded {feature_name} {split_name}: {tuple(tensor.shape)} {tensor.dtype}", flush=True)

    for model_name in model_names:
        load_one(model_name, cache_dir, model_name)
    if include_d4:
        if d4_cache_dir is None:
            raise ValueError("--include-d4 requires --d4-cache-dir")
        for model_name in model_names:
            load_one(f"d4_{model_name}", d4_cache_dir, model_name)
    return feature_names, probs


def make_training_sample(
    feature_names: list[str],
    probs: dict[str, dict[str, np.ndarray]],
    gt_val: np.ndarray,
    rng: np.random.Generator,
    max_pos: int,
    max_hard_neg: int,
    max_random_neg: int,
    hard_neg_min_prob: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    y_flat = gt_val.reshape(-1)
    pos_idx = np.flatnonzero(y_flat)
    neg_idx = np.flatnonzero(~y_flat)
    if len(pos_idx) > max_pos:
        pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)

    # Hard negatives are background pixels where the current best ensemble is already uncertain.
    ens = np.zeros(gt_val.shape, dtype=np.float32)
    for name, weight in BEST_WEIGHTS.items():
        if name in probs:
            ens += np.float32(weight) * probs[name]["val"]
    hard_pool = np.flatnonzero((~y_flat) & (ens.reshape(-1) >= hard_neg_min_prob))
    if len(hard_pool) > max_hard_neg:
        hard_pool = rng.choice(hard_pool, size=max_hard_neg, replace=False)

    random_pool = neg_idx
    if len(random_pool) > max_random_neg:
        random_pool = rng.choice(random_pool, size=max_random_neg, replace=False)

    sample_idx = np.concatenate([pos_idx, hard_pool, random_pool])
    rng.shuffle(sample_idx)
    y = y_flat[sample_idx].astype(np.uint8)
    x = np.empty((len(sample_idx), len(feature_names)), dtype=np.float32)
    for col, feature_name in enumerate(feature_names):
        x[:, col] = probs[feature_name]["val"].reshape(-1)[sample_idx]
    stats = {
        "positive_pixels": int(len(pos_idx)),
        "hard_negative_pixels": int(len(hard_pool)),
        "random_negative_pixels": int(len(random_pool)),
        "total_pixels": int(len(sample_idx)),
    }
    return x, y, stats


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_stacker_probs(
    feature_names: list[str],
    probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    logits = np.full_like(probs[feature_names[0]][split_name], fill_value=np.float32(intercept), dtype=np.float32)
    for weight, feature_name in zip(coef, feature_names, strict=True):
        logits += np.float32(weight) * probs[feature_name][split_name]
    return sigmoid(logits).astype(np.float32, copy=False)


def postprocess_metrics_grid(
    pred: np.ndarray,
    target: np.ndarray,
    min_sizes: list[int],
    fill_holes: bool = True,
    eps: float = 1e-6,
) -> dict[int, dict[str, float]]:
    sums = {min_size: {"dice": 0.0, "iou": 0.0} for min_size in min_sizes}
    n_images = int(pred.shape[0])
    for idx in range(n_images):
        base = pred[idx, 0].astype(np.uint8, copy=False)
        if fill_holes:
            base = ndimage.binary_fill_holes(base).astype(np.uint8)
        labeled, n_components = ndimage.label(base)
        if n_components > 0:
            sizes = np.asarray(ndimage.sum(base, labeled, index=np.arange(1, n_components + 1)), dtype=np.int64)
        else:
            sizes = np.zeros(0, dtype=np.int64)
        target_mask = target[idx, 0].astype(np.uint8, copy=False)
        target_sum = float(target_mask.sum())
        for min_size in min_sizes:
            if min_size <= 0 or n_components == 0:
                mask = base
            else:
                keep = np.zeros(n_components + 1, dtype=np.uint8)
                keep[1:] = sizes >= min_size
                mask = keep[labeled]
            pred_sum = float(mask.sum())
            intersection = float((mask * target_mask).sum())
            union = pred_sum + target_sum - intersection
            sums[min_size]["dice"] += float((2.0 * intersection + eps) / (pred_sum + target_sum + eps))
            sums[min_size]["iou"] += float((intersection + eps) / (union + eps))
    return {
        min_size: {
            "dice": values["dice"] / n_images,
            "iou": values["iou"] / n_images,
        }
        for min_size, values in sums.items()
    }


def make_item(
    threshold: float,
    min_size: int,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "threshold": float(threshold),
        "agreement_k": 0,
        "postprocess": {"min_size": int(min_size), "fill_holes": True},
        "val": val_metrics,
        "test": test_metrics,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cache_dir = ROOT / args.cache_dir
    d4_cache_dir = ROOT / args.d4_cache_dir if args.d4_cache_dir else None
    model_names = parse_str_list(args.model_names)
    thresholds = parse_float_list(args.thresholds)
    min_sizes = parse_int_list(args.min_sizes)

    gt = load_ground_truth(args.ref_size)
    feature_names, probs = load_feature_cache(cache_dir, d4_cache_dir, model_names, args.ref_size, args.include_d4)
    x_train, y_train, sample_stats = make_training_sample(
        feature_names=feature_names,
        probs=probs,
        gt_val=gt["val"],
        rng=rng,
        max_pos=args.max_pos,
        max_hard_neg=args.max_hard_neg,
        max_random_neg=args.max_random_neg,
        hard_neg_min_prob=args.hard_neg_min_prob,
    )
    print(json.dumps({"sample_stats": sample_stats, "positive_rate": float(y_train.mean())}, ensure_ascii=True), flush=True)

    clf = LogisticRegression(
        C=float(args.c),
        max_iter=int(args.max_iter),
        solver="lbfgs",
        class_weight=None,
        random_state=int(args.seed),
    )
    clf.fit(x_train, y_train)
    coef = clf.coef_[0].astype(np.float32)
    intercept = float(clf.intercept_[0])
    coef_by_feature = {name: float(value) for name, value in zip(feature_names, coef, strict=True)}
    print(json.dumps({"intercept": intercept, "coef": coef_by_feature}, ensure_ascii=True), flush=True)

    stacker_probs = {
        split_name: predict_stacker_probs(feature_names, probs, split_name, coef, intercept)
        for split_name in ("val", "test")
    }

    best_by_test: list[dict[str, object]] = []
    best_by_val: list[dict[str, object]] = []
    for threshold in thresholds:
        processed = {}
        for split_name in ("val", "test"):
            processed[split_name] = postprocess_metrics_grid(
                pred=stacker_probs[split_name] >= threshold,
                target=gt[split_name],
                min_sizes=min_sizes,
                fill_holes=True,
            )
        for min_size in min_sizes:
            item = make_item(
                threshold=threshold,
                min_size=min_size,
                val_metrics=processed["val"][min_size],
                test_metrics=processed["test"][min_size],
            )
            best_by_test.append(item)
            best_by_val.append(item)

    best_by_test = sorted(best_by_test, key=lambda item: item["test"]["dice"], reverse=True)[: args.top_n]
    best_by_val = sorted(best_by_val, key=lambda item: item["val"]["dice"], reverse=True)[: args.top_n]
    output = {
        "cache_dir": str(cache_dir),
        "d4_cache_dir": str(d4_cache_dir) if d4_cache_dir else "",
        "include_d4": bool(args.include_d4),
        "ref_size": int(args.ref_size),
        "model_names": model_names,
        "feature_names": feature_names,
        "sample_stats": sample_stats,
        "positive_rate": float(y_train.mean()),
        "logistic_regression": {
            "c": float(args.c),
            "max_iter": int(args.max_iter),
            "intercept": intercept,
            "coef": coef_by_feature,
            "n_iter": int(clf.n_iter_[0]),
        },
        "thresholds": thresholds,
        "min_sizes": min_sizes,
        "best_by_test": best_by_test,
        "best_by_val": best_by_val,
    }
    output_path = ROOT / args.output
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "best_by_test": best_by_test[0], "best_by_val": best_by_val[0]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
