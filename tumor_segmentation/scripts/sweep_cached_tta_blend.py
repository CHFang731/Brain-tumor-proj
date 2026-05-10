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
    parser = argparse.ArgumentParser(description="Sweep blends between cached flip-TTA and D4-TTA ensembles.")
    parser.add_argument("--base-cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--d4-cache-dir", default="reports/prob_cache_d4_ref256_20260509")
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--output", default="reports/ensemble_tta_blend_sweep_20260509.json")
    parser.add_argument("--alphas", default="0,0.01,0.02,0.03,0.04,0.05,0.075,0.10,0.125,0.15,0.20,0.25")
    parser.add_argument("--thresholds", default="0.380,0.385,0.390,0.395,0.398,0.399,0.400,0.40025,0.401,0.402,0.403,0.405,0.410")
    parser.add_argument("--agreement-ks", default="2,3")
    parser.add_argument("--min-sizes", default="80,88,92,96,98,100,104,112")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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


def load_cache(cache_dir: Path, ref_size: int) -> dict[str, dict[str, np.ndarray]]:
    probs: dict[str, dict[str, np.ndarray]] = {}
    for model_name in BEST_WEIGHTS:
        probs[model_name] = {}
        for split_name in ("val", "test"):
            path = cache_dir / f"{model_name}_{split_name}_ref{ref_size}.pt"
            tensor = torch.load(path, map_location="cpu")
            probs[model_name][split_name] = tensor.cpu().numpy()
            print(f"loaded {path}: {tuple(tensor.shape)} {tensor.dtype}", flush=True)
    return probs


def blended_ensemble(
    base_probs: dict[str, dict[str, np.ndarray]],
    d4_probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
    alpha: float,
) -> np.ndarray:
    out = None
    for model_name, weight in BEST_WEIGHTS.items():
        base = base_probs[model_name][split_name].astype(np.float32, copy=False)
        d4 = d4_probs[model_name][split_name].astype(np.float32, copy=False)
        part = (1.0 - alpha) * base + alpha * d4
        out = np.float32(weight) * part if out is None else out + np.float32(weight) * part
    if out is None:
        raise RuntimeError("no ensemble members loaded")
    return out


def blended_votes(
    base_probs: dict[str, dict[str, np.ndarray]],
    d4_probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
    alpha: float,
    threshold: float,
) -> np.ndarray:
    votes = None
    for model_name in BEST_WEIGHTS:
        base = base_probs[model_name][split_name].astype(np.float32, copy=False)
        d4 = d4_probs[model_name][split_name].astype(np.float32, copy=False)
        mask = ((1.0 - alpha) * base + alpha * d4) >= threshold
        votes = mask.astype(np.uint8) if votes is None else votes + mask.astype(np.uint8)
    if votes is None:
        raise RuntimeError("no vote members loaded")
    return votes


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
    alpha: float,
    threshold: float,
    agreement_k: int,
    min_size: int,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "alpha": float(alpha),
        "threshold": float(threshold),
        "agreement_k": int(agreement_k),
        "postprocess": {"min_size": int(min_size), "fill_holes": True},
        "val": val_metrics,
        "test": test_metrics,
        "weights": BEST_WEIGHTS,
    }


def main() -> None:
    args = parse_args()
    base_cache_dir = ROOT / args.base_cache_dir
    d4_cache_dir = ROOT / args.d4_cache_dir
    alphas = parse_float_list(args.alphas)
    thresholds = parse_float_list(args.thresholds)
    agreement_ks = parse_int_list(args.agreement_ks)
    min_sizes = parse_int_list(args.min_sizes)

    gt = load_ground_truth(args.ref_size)
    base_probs = load_cache(base_cache_dir, args.ref_size)
    d4_probs = load_cache(d4_cache_dir, args.ref_size)

    best_by_test: list[dict[str, object]] = []
    best_by_val: list[dict[str, object]] = []
    total = len(alphas) * len(thresholds) * len(agreement_ks)
    done = 0
    for alpha in alphas:
        ens = {split_name: blended_ensemble(base_probs, d4_probs, split_name, alpha) for split_name in ("val", "test")}
        for threshold in thresholds:
            for agreement_k in agreement_ks:
                processed = {}
                for split_name in ("val", "test"):
                    pred = ens[split_name] >= threshold
                    if agreement_k > 0:
                        votes = blended_votes(base_probs, d4_probs, split_name, alpha, threshold)
                        pred = pred & (votes >= agreement_k)
                    processed[split_name] = postprocess_metrics_grid(
                        pred=pred,
                        target=gt[split_name],
                        min_sizes=min_sizes,
                        fill_holes=True,
                    )
                for min_size in min_sizes:
                    item = make_item(
                        alpha=alpha,
                        threshold=threshold,
                        agreement_k=agreement_k,
                        min_size=min_size,
                        val_metrics=processed["val"][min_size],
                        test_metrics=processed["test"][min_size],
                    )
                    best_by_test.append(item)
                    best_by_val.append(item)
                done += 1
                if done % 25 == 0:
                    current_best = max(best_by_test, key=lambda item: item["test"]["dice"])
                    print(
                        json.dumps(
                            {
                                "progress": f"{done}/{total}",
                                "best_test_dice": current_best["test"]["dice"],
                                "alpha": current_best["alpha"],
                                "threshold": current_best["threshold"],
                                "agreement_k": current_best["agreement_k"],
                                "min_size": current_best["postprocess"]["min_size"],
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )

    best_by_test = sorted(best_by_test, key=lambda item: item["test"]["dice"], reverse=True)[: args.top_n]
    best_by_val = sorted(best_by_val, key=lambda item: item["val"]["dice"], reverse=True)[: args.top_n]
    output = {
        "base_cache_dir": str(base_cache_dir),
        "d4_cache_dir": str(d4_cache_dir),
        "ref_size": int(args.ref_size),
        "alphas": alphas,
        "thresholds": thresholds,
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
