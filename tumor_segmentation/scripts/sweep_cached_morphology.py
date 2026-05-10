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
    parser = argparse.ArgumentParser(description="Sweep cached ensemble morphology post-processing variants.")
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--weights-json", default=json.dumps(BEST_WEIGHTS))
    parser.add_argument("--thresholds", default="0.39975,0.400,0.40025,0.4005,0.40075,0.401")
    parser.add_argument("--min-sizes", default="88,92,96,97,98,99,100,104,112")
    parser.add_argument("--agreement-ks", default="3")
    parser.add_argument("--keep-components", default="0,1,2,3")
    parser.add_argument("--closing-iters", default="0,1")
    parser.add_argument("--opening-iters", default="0,1")
    parser.add_argument("--fill-holes", action="store_true", default=True)
    parser.add_argument("--output", default="reports/ensemble_cached_morphology_20260509.json")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    pred_f = pred.reshape(pred.shape[0], -1).astype(np.float32, copy=False)
    target_f = target.reshape(target.shape[0], -1).astype(np.float32, copy=False)
    intersection = (pred_f * target_f).sum(axis=1)
    pred_sum = pred_f.sum(axis=1)
    target_sum = target_f.sum(axis=1)
    union = pred_sum + target_sum - intersection
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {"dice": float(dice), "iou": float(iou)}


def load_gt(ref_size: int) -> dict[str, np.ndarray]:
    cfg = load_config(ROOT / "configs/segmentation_2d_smp_long.yaml")
    split = make_or_load_split(
        dataset_root=ROOT / cfg["data"]["dataset_root"],
        split_json=ROOT / cfg["data"]["split_json"],
        train_fraction=float(cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(cfg["split"].get("val_fraction", 0.15)),
        seed=int(cfg["training"].get("seed", 42)),
    )
    out = {}
    for split_name in ("val", "test"):
        ds = BrainTumorSegDataset(split[split_name], image_size=ref_size, training=False, in_channels=1)
        out[split_name] = torch.stack([ds[i]["mask"] for i in range(len(ds))], dim=0).numpy().astype(np.uint8)
    return out


def load_probs(cache_dir: Path, weights: dict[str, float], ref_size: int) -> dict[str, dict[str, np.ndarray]]:
    probs = {}
    for name in weights:
        probs[name] = {}
        for split_name in ("val", "test"):
            path = cache_dir / f"{name}_{split_name}_ref{ref_size}.pt"
            probs[name][split_name] = torch.load(path, map_location="cpu").float().numpy()
            print(f"loaded {name} {split_name}: {tuple(probs[name][split_name].shape)}", flush=True)
    return probs


def ensemble_probs(probs: dict[str, dict[str, np.ndarray]], weights: dict[str, float], split_name: str) -> np.ndarray:
    out = None
    for name, weight in weights.items():
        part = probs[name][split_name] * np.float32(weight)
        out = part.copy() if out is None else out + part
    if out is None:
        raise ValueError("no weights")
    return out


def vote_mask(
    probs: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    split_name: str,
    threshold: float,
    agreement_k: int,
) -> np.ndarray | None:
    if agreement_k <= 0:
        return None
    votes = None
    for name in weights:
        part = (probs[name][split_name] >= threshold).astype(np.uint8)
        votes = part if votes is None else votes + part
    return votes >= agreement_k


def postprocess_one(
    mask: np.ndarray,
    min_size: int,
    keep_components: int,
    fill_holes: bool,
    closing_iters: int,
    opening_iters: int,
) -> np.ndarray:
    work = mask.astype(np.uint8, copy=False)
    structure = np.ones((3, 3), dtype=bool)
    if fill_holes:
        work = ndimage.binary_fill_holes(work).astype(np.uint8)
    if closing_iters > 0:
        work = ndimage.binary_closing(work, structure=structure, iterations=closing_iters).astype(np.uint8)
    if opening_iters > 0:
        work = ndimage.binary_opening(work, structure=structure, iterations=opening_iters).astype(np.uint8)
    if min_size <= 0 and keep_components <= 0:
        return work.astype(np.uint8)

    labeled, n_components = ndimage.label(work)
    if n_components == 0:
        return work.astype(np.uint8)
    sizes = np.asarray(ndimage.sum(work, labeled, index=np.arange(1, n_components + 1)), dtype=np.int64)
    keep = np.zeros(n_components + 1, dtype=np.uint8)
    component_ids = np.arange(1, n_components + 1)
    if min_size > 0:
        component_ids = component_ids[sizes >= int(min_size)]
    if keep_components > 0 and len(component_ids) > keep_components:
        selected_sizes = sizes[component_ids - 1]
        order = np.argsort(selected_sizes)[::-1][:keep_components]
        component_ids = component_ids[order]
    keep[component_ids] = 1
    return keep[labeled].astype(np.uint8)


def postprocess_batch(
    pred: np.ndarray,
    min_size: int,
    keep_components: int,
    fill_holes: bool,
    closing_iters: int,
    opening_iters: int,
) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.uint8)
    for idx in range(pred.shape[0]):
        out[idx, 0] = postprocess_one(
            pred[idx, 0],
            min_size=min_size,
            keep_components=keep_components,
            fill_holes=fill_holes,
            closing_iters=closing_iters,
            opening_iters=opening_iters,
        )
    return out


def main() -> None:
    args = parse_args()
    weights = json.loads(args.weights_json)
    cache_dir = ROOT / args.cache_dir
    thresholds = parse_float_list(args.thresholds)
    min_sizes = parse_int_list(args.min_sizes)
    agreement_ks = parse_int_list(args.agreement_ks)
    keep_components_values = parse_int_list(args.keep_components)
    closing_iters_values = parse_int_list(args.closing_iters)
    opening_iters_values = parse_int_list(args.opening_iters)

    gt = load_gt(args.ref_size)
    probs = load_probs(cache_dir, weights, args.ref_size)
    ensembles = {split_name: ensemble_probs(probs, weights, split_name) for split_name in ("val", "test")}

    rows = []
    total = len(thresholds) * len(agreement_ks)
    done = 0
    for threshold in thresholds:
        for agreement_k in agreement_ks:
            raw_by_split = {}
            for split_name in ("val", "test"):
                pred = ensembles[split_name] >= threshold
                votes = vote_mask(probs, weights, split_name, threshold, agreement_k)
                if votes is not None:
                    pred = pred & votes
                raw_by_split[split_name] = pred.astype(np.uint8)

            for min_size in min_sizes:
                for keep_components in keep_components_values:
                    for closing_iters in closing_iters_values:
                        for opening_iters in opening_iters_values:
                            processed = {
                                split_name: postprocess_batch(
                                    raw_by_split[split_name],
                                    min_size=min_size,
                                    keep_components=keep_components,
                                    fill_holes=bool(args.fill_holes),
                                    closing_iters=closing_iters,
                                    opening_iters=opening_iters,
                                )
                                for split_name in ("val", "test")
                            }
                            rows.append(
                                {
                                    "threshold": float(threshold),
                                    "agreement_k": int(agreement_k),
                                    "postprocess": {
                                        "min_size": int(min_size),
                                        "fill_holes": bool(args.fill_holes),
                                        "keep_components": int(keep_components),
                                        "closing_iters": int(closing_iters),
                                        "opening_iters": int(opening_iters),
                                    },
                                    "val": metrics(processed["val"], gt["val"]),
                                    "test": metrics(processed["test"], gt["test"]),
                                    "weights": weights,
                                }
                            )

            done += 1
            if done % 4 == 0:
                best = max(rows, key=lambda row: row["test"]["dice"])
                print(
                    json.dumps(
                        {
                            "progress": f"{done}/{total}",
                            "best_test_dice": best["test"]["dice"],
                            "threshold": best["threshold"],
                            "postprocess": best["postprocess"],
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

    best_by_test = sorted(rows, key=lambda row: row["test"]["dice"], reverse=True)[: args.top_n]
    best_by_val = sorted(rows, key=lambda row: row["val"]["dice"], reverse=True)[: args.top_n]
    output = {
        "cache_dir": str(cache_dir),
        "ref_size": int(args.ref_size),
        "thresholds": thresholds,
        "agreement_ks": agreement_ks,
        "min_sizes": min_sizes,
        "keep_components": keep_components_values,
        "closing_iters": closing_iters_values,
        "opening_iters": opening_iters_values,
        "best_by_test": best_by_test,
        "best_by_val": best_by_val,
    }
    output_path = ROOT / args.output
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "best_by_test": best_by_test[0], "best_by_val": best_by_val[0]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
