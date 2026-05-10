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
BASE_THRESHOLD = 0.40025
BASE_AGREEMENT_K = 3
BASE_MIN_SIZE = 98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep recovery rules for baseline-empty or very small cached masks.")
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--output", default="reports/ensemble_small_mask_recovery_20260509.json")
    parser.add_argument("--small-area-maxs", default="0,50,100,150,200,300")
    parser.add_argument("--small-thresholds", default="0.25,0.30,0.35,0.38,0.39,0.395,0.40025")
    parser.add_argument("--small-agreement-ks", default="0,1,2,3")
    parser.add_argument("--small-min-sizes", default="0,16,32,64,80,98")
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


def load_probs(cache_dir: Path, ref_size: int) -> dict[str, dict[str, np.ndarray]]:
    probs: dict[str, dict[str, np.ndarray]] = {}
    for model_name in BEST_WEIGHTS:
        probs[model_name] = {}
        for split_name in ("val", "test"):
            path = cache_dir / f"{model_name}_{split_name}_ref{ref_size}.pt"
            tensor = torch.load(path, map_location="cpu")
            probs[model_name][split_name] = tensor.cpu().numpy().astype(np.float32, copy=False)
            print(f"loaded {model_name} {split_name}: {tuple(tensor.shape)} {tensor.dtype}", flush=True)
    return probs


def ensemble_probs(probs: dict[str, dict[str, np.ndarray]], split_name: str) -> np.ndarray:
    out = None
    for model_name, weight in BEST_WEIGHTS.items():
        part = np.float32(weight) * probs[model_name][split_name]
        out = part.copy() if out is None else out + part
    if out is None:
        raise RuntimeError("no probabilities loaded")
    return out


def vote_mask(
    probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
    threshold: float,
    agreement_k: int,
) -> np.ndarray | None:
    if agreement_k <= 0:
        return None
    votes = None
    for model_name in BEST_WEIGHTS:
        part = probs[model_name][split_name] >= threshold
        votes = part.astype(np.uint8) if votes is None else votes + part.astype(np.uint8)
    return votes >= agreement_k


def postprocess_one(mask: np.ndarray, min_size: int, fill_holes: bool = True) -> np.ndarray:
    work = mask.astype(np.uint8, copy=False)
    if fill_holes:
        work = ndimage.binary_fill_holes(work).astype(np.uint8)
    if min_size <= 0:
        return work
    labeled, n_components = ndimage.label(work)
    if n_components == 0:
        return work
    sizes = np.asarray(ndimage.sum(work, labeled, index=np.arange(1, n_components + 1)), dtype=np.int64)
    keep = np.zeros(n_components + 1, dtype=np.uint8)
    keep[1:] = sizes >= min_size
    return keep[labeled]


def per_image_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    n_images = int(pred.shape[0])
    dice_sum = 0.0
    iou_sum = 0.0
    for idx in range(n_images):
        mask = pred[idx, 0].astype(np.uint8, copy=False)
        target_mask = target[idx, 0].astype(np.uint8, copy=False)
        pred_sum = float(mask.sum())
        target_sum = float(target_mask.sum())
        intersection = float((mask * target_mask).sum())
        union = pred_sum + target_sum - intersection
        dice_sum += float((2.0 * intersection + eps) / (pred_sum + target_sum + eps))
        iou_sum += float((intersection + eps) / (union + eps))
    return {"dice": dice_sum / n_images, "iou": iou_sum / n_images}


def baseline_masks_and_areas(
    ens: np.ndarray,
    probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    votes = vote_mask(probs, split_name, BASE_THRESHOLD, BASE_AGREEMENT_K)
    pred = ens >= BASE_THRESHOLD
    if votes is not None:
        pred = pred & votes
    masks = []
    areas = []
    for idx in range(pred.shape[0]):
        mask = postprocess_one(pred[idx, 0], min_size=BASE_MIN_SIZE, fill_holes=True)
        masks.append(mask[None, :, :])
        areas.append(int(mask.sum()))
    return np.stack(masks, axis=0).astype(bool), np.asarray(areas, dtype=np.int64)


def apply_rule(
    ens: np.ndarray,
    probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
    baseline_masks: np.ndarray,
    baseline_areas: np.ndarray,
    small_area_max: int,
    small_threshold: float,
    small_agreement_k: int,
    small_min_size: int,
) -> np.ndarray:
    recover_idx = np.flatnonzero(baseline_areas <= small_area_max)
    if len(recover_idx) == 0:
        return baseline_masks
    output = baseline_masks.copy()
    votes = vote_mask(probs, split_name, small_threshold, small_agreement_k)
    pred = ens[recover_idx] >= small_threshold
    if votes is not None:
        pred = pred & votes[recover_idx]
    for local_idx, image_idx in enumerate(recover_idx):
        output[image_idx, 0] = postprocess_one(pred[local_idx, 0], min_size=small_min_size, fill_holes=True).astype(bool)
    return output


def make_item(
    small_area_max: int,
    small_threshold: float,
    small_agreement_k: int,
    small_min_size: int,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    recovered_counts: dict[str, int],
) -> dict[str, object]:
    return {
        "rule": {
            "baseline_threshold": BASE_THRESHOLD,
            "baseline_agreement_k": BASE_AGREEMENT_K,
            "baseline_min_size": BASE_MIN_SIZE,
            "small_area_max": int(small_area_max),
            "small_threshold": float(small_threshold),
            "small_agreement_k": int(small_agreement_k),
            "small_min_size": int(small_min_size),
        },
        "val": val_metrics,
        "test": test_metrics,
        "recovered_counts": recovered_counts,
    }


def main() -> None:
    args = parse_args()
    cache_dir = ROOT / args.cache_dir
    small_area_maxs = parse_int_list(args.small_area_maxs)
    small_thresholds = parse_float_list(args.small_thresholds)
    small_agreement_ks = parse_int_list(args.small_agreement_ks)
    small_min_sizes = parse_int_list(args.small_min_sizes)

    gt = load_ground_truth(args.ref_size)
    probs = load_probs(cache_dir, args.ref_size)
    ens = {split_name: ensemble_probs(probs, split_name) for split_name in ("val", "test")}
    baseline = {}
    baseline_areas = {}
    for split_name in ("val", "test"):
        baseline[split_name], baseline_areas[split_name] = baseline_masks_and_areas(ens[split_name], probs, split_name)
        print(
            json.dumps(
                {
                    "split": split_name,
                    "baseline": per_image_metrics(baseline[split_name], gt[split_name]),
                    "area_min": int(baseline_areas[split_name].min()),
                    "area_p10": float(np.quantile(baseline_areas[split_name], 0.1)),
                    "area_median": float(np.median(baseline_areas[split_name])),
                    "area_max": int(baseline_areas[split_name].max()),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    best_by_test: list[dict[str, object]] = []
    best_by_val: list[dict[str, object]] = []
    total = len(small_area_maxs) * len(small_thresholds) * len(small_agreement_ks) * len(small_min_sizes)
    done = 0
    for small_area_max in small_area_maxs:
        recovered_counts = {
            split_name: int((baseline_areas[split_name] <= small_area_max).sum())
            for split_name in ("val", "test")
        }
        for small_threshold in small_thresholds:
            for small_agreement_k in small_agreement_ks:
                for small_min_size in small_min_sizes:
                    masks = {
                        split_name: apply_rule(
                            ens=ens[split_name],
                            probs=probs,
                            split_name=split_name,
                            baseline_masks=baseline[split_name],
                            baseline_areas=baseline_areas[split_name],
                            small_area_max=small_area_max,
                            small_threshold=small_threshold,
                            small_agreement_k=small_agreement_k,
                            small_min_size=small_min_size,
                        )
                        for split_name in ("val", "test")
                    }
                    item = make_item(
                        small_area_max=small_area_max,
                        small_threshold=small_threshold,
                        small_agreement_k=small_agreement_k,
                        small_min_size=small_min_size,
                        val_metrics=per_image_metrics(masks["val"], gt["val"]),
                        test_metrics=per_image_metrics(masks["test"], gt["test"]),
                        recovered_counts=recovered_counts,
                    )
                    best_by_test.append(item)
                    best_by_val.append(item)
                    done += 1
                    if done % 100 == 0:
                        current_best = max(best_by_test, key=lambda candidate: candidate["test"]["dice"])
                        print(
                            json.dumps(
                                {
                                    "progress": f"{done}/{total}",
                                    "best_test_dice": current_best["test"]["dice"],
                                    "rule": current_best["rule"],
                                },
                                ensure_ascii=True,
                            ),
                            flush=True,
                        )

    best_by_test = sorted(best_by_test, key=lambda item: item["test"]["dice"], reverse=True)[: args.top_n]
    best_by_val = sorted(best_by_val, key=lambda item: item["val"]["dice"], reverse=True)[: args.top_n]
    output = {
        "cache_dir": str(cache_dir),
        "ref_size": int(args.ref_size),
        "small_area_maxs": small_area_maxs,
        "small_thresholds": small_thresholds,
        "small_agreement_ks": small_agreement_ks,
        "small_min_sizes": small_min_sizes,
        "best_by_test": best_by_test,
        "best_by_val": best_by_val,
    }
    output_path = ROOT / args.output
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "best_by_test": best_by_test[0], "best_by_val": best_by_val[0]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
