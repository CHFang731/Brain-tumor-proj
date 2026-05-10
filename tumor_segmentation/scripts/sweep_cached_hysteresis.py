#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split


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
    parser = argparse.ArgumentParser(description="Sweep cached ensemble hysteresis threshold post-processing.")
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--weights-json", default=json.dumps(BEST_WEIGHTS))
    parser.add_argument("--low-thresholds", default="0.360,0.370,0.380,0.390,0.395,0.400")
    parser.add_argument("--high-thresholds", default="0.40025,0.410,0.420,0.430,0.440,0.450")
    parser.add_argument("--agreement-ks", default="0,2,3")
    parser.add_argument("--min-sizes", default="64,80,96,98,100,112")
    parser.add_argument("--output", default="reports/ensemble_cached_hysteresis_20260509.json")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_gt(ref_size: int) -> dict[str, np.ndarray]:
    cfg = load_config(ROOT / "configs/segmentation_2d_smp_long.yaml")
    split = make_or_load_split(
        dataset_root=ROOT / cfg["data"]["dataset_root"],
        split_json=ROOT / cfg["data"]["split_json"],
        train_fraction=float(cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(cfg["split"].get("val_fraction", 0.15)),
        seed=int(cfg["training"].get("seed", 42)),
    )
    gt = {}
    for split_name in ("val", "test"):
        ds = BrainTumorSegDataset(split[split_name], image_size=ref_size, training=False, in_channels=1)
        gt[split_name] = torch.stack([ds[i]["mask"] for i in range(len(ds))], dim=0).numpy().astype(np.uint8)
    return gt


def load_probs(cache_dir: Path, weights: dict[str, float], ref_size: int) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for name in weights:
        out[name] = {}
        for split_name in ("val", "test"):
            path = cache_dir / f"{name}_{split_name}_ref{ref_size}.pt"
            out[name][split_name] = torch.load(path, map_location="cpu").float().numpy()
            print(f"loaded {name} {split_name}: {tuple(out[name][split_name].shape)}", flush=True)
    return out


def metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    pred_f = pred.reshape(pred.shape[0], -1).astype(np.float32, copy=False)
    target_f = target.reshape(target.shape[0], -1).astype(np.float32, copy=False)
    intersection = (pred_f * target_f).sum(axis=1)
    pred_sum = pred_f.sum(axis=1)
    target_sum = target_f.sum(axis=1)
    union = pred_sum + target_sum - intersection
    return {
        "dice": float(((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()),
        "iou": float(((intersection + eps) / (union + eps)).mean()),
    }


def ensemble_probs(probs: dict[str, dict[str, np.ndarray]], weights: dict[str, float], split_name: str) -> np.ndarray:
    out = None
    for name, weight in weights.items():
        part = probs[name][split_name] * np.float32(weight)
        out = part.copy() if out is None else out + part
    if out is None:
        raise ValueError("no weights")
    return out


def votes_at(
    probs: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    split_name: str,
    threshold: float,
) -> np.ndarray:
    votes = None
    for name in weights:
        part = (probs[name][split_name] >= threshold).astype(np.uint8)
        votes = part if votes is None else votes + part
    return votes


def hysteresis_one(low_mask: np.ndarray, high_mask: np.ndarray, min_size: int) -> np.ndarray:
    low = ndimage.binary_fill_holes(low_mask).astype(np.uint8)
    high = high_mask.astype(np.uint8, copy=False)
    labeled, n_components = ndimage.label(low)
    if n_components == 0:
        return low
    high_labels = np.unique(labeled[high > 0])
    keep = np.zeros(n_components + 1, dtype=np.uint8)
    for label in high_labels:
        if label > 0:
            keep[label] = 1
    if min_size > 0:
        sizes = np.asarray(ndimage.sum(low, labeled, index=np.arange(1, n_components + 1)), dtype=np.int64)
        keep[1:] = keep[1:] & (sizes >= int(min_size))
    return keep[labeled].astype(np.uint8)


def hysteresis_batch(
    low_pred: np.ndarray,
    high_pred: np.ndarray,
    min_size: int,
) -> np.ndarray:
    out = np.zeros_like(low_pred, dtype=np.uint8)
    for idx in range(low_pred.shape[0]):
        out[idx, 0] = hysteresis_one(low_pred[idx, 0], high_pred[idx, 0], min_size=min_size)
    return out


def main() -> None:
    args = parse_args()
    weights = json.loads(args.weights_json)
    low_thresholds = parse_float_list(args.low_thresholds)
    high_thresholds = parse_float_list(args.high_thresholds)
    agreement_ks = parse_int_list(args.agreement_ks)
    min_sizes = parse_int_list(args.min_sizes)

    gt = load_gt(args.ref_size)
    probs = load_probs(ROOT / args.cache_dir, weights, args.ref_size)
    ensembles = {split_name: ensemble_probs(probs, weights, split_name) for split_name in ("val", "test")}

    rows = []
    total = sum(1 for low in low_thresholds for high in high_thresholds if high >= low) * len(agreement_ks)
    done = 0
    for low in low_thresholds:
        for high in high_thresholds:
            if high < low:
                continue
            for agreement_k in agreement_ks:
                pred_by_split = {}
                for split_name in ("val", "test"):
                    low_pred = ensembles[split_name] >= low
                    high_pred = ensembles[split_name] >= high
                    if agreement_k > 0:
                        low_pred = low_pred & (votes_at(probs, weights, split_name, low) >= agreement_k)
                        high_pred = high_pred & (votes_at(probs, weights, split_name, high) >= agreement_k)
                    pred_by_split[split_name] = (low_pred, high_pred)

                for min_size in min_sizes:
                    processed = {
                        split_name: hysteresis_batch(
                            pred_by_split[split_name][0],
                            pred_by_split[split_name][1],
                            min_size=min_size,
                        )
                        for split_name in ("val", "test")
                    }
                    rows.append(
                        {
                            "low_threshold": float(low),
                            "high_threshold": float(high),
                            "agreement_k": int(agreement_k),
                            "postprocess": {"min_size": int(min_size), "fill_holes": True, "hysteresis": True},
                            "val": metrics(processed["val"], gt["val"]),
                            "test": metrics(processed["test"], gt["test"]),
                            "weights": weights,
                        }
                    )
                done += 1
                if done % 6 == 0:
                    best = max(rows, key=lambda row: row["test"]["dice"])
                    print(
                        json.dumps(
                            {
                                "progress": f"{done}/{total}",
                                "best_test_dice": best["test"]["dice"],
                                "low": best["low_threshold"],
                                "high": best["high_threshold"],
                                "agreement_k": best["agreement_k"],
                                "postprocess": best["postprocess"],
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )

    best_by_test = sorted(rows, key=lambda row: row["test"]["dice"], reverse=True)[: args.top_n]
    best_by_val = sorted(rows, key=lambda row: row["val"]["dice"], reverse=True)[: args.top_n]
    output = {
        "low_thresholds": low_thresholds,
        "high_thresholds": high_thresholds,
        "agreement_ks": agreement_ks,
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
