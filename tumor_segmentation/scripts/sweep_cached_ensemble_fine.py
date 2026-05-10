#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split


BEST_BASE = {
    "old": 0.12698717301282703,
    "s44": 0.1338425658410686,
    "s45": 0.04212298739392233,
    "effb4ft": 0.158983941016059,
    "s384": 0.2206164260063713,
    "s384s45": 0.158983941016059,
    "s384s56": 0.14836297581368268,
    "s384s64": 0.010099989900010101,
}


@dataclass(frozen=True)
class Candidate:
    tag: str
    weights: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine sweep cached segmentation ensemble candidates.")
    parser.add_argument("--cache-dir", default="reports/prob_cache_ref256_20260507")
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--output", default="reports/ensemble_cached_fine_sweep_20260507.json")
    parser.add_argument("--thresholds", default="0.400,0.401,0.402,0.403,0.404,0.405,0.406,0.407,0.4075,0.408,0.409,0.410,0.411,0.412,0.413,0.414,0.415")
    parser.add_argument("--min-sizes", default="76,80,84,86,88,90,92,96,100,104,108,112")
    parser.add_argument("--agreement-ks", default="3")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on generated candidates for smoke tests.")
    parser.add_argument("--eff-min", type=float, default=0.115)
    parser.add_argument("--eff-max", type=float, default=0.165)
    parser.add_argument("--eff-step", type=float, default=0.0025)
    parser.add_argument("--old-min", type=float, default=0.105)
    parser.add_argument("--old-max", type=float, default=0.165)
    parser.add_argument("--old-step", type=float, default=0.005)
    parser.add_argument("--s45-min", type=float, default=0.020)
    parser.add_argument("--s45-max", type=float, default=0.065)
    parser.add_argument("--s45-step", type=float, default=0.005)
    parser.add_argument("--s384s45-min", type=float, default=0.140)
    parser.add_argument("--s384s45-max", type=float, default=0.205)
    parser.add_argument("--s384s45-step", type=float, default=0.005)
    parser.add_argument("--extra-names", default="s384s62,s384s65,s384s66,s384s67,s384s69,s384s64,s384s50c,s384s60,s45c,s57,s56,s50")
    parser.add_argument("--extra-min", type=float, default=0.005)
    parser.add_argument("--extra-max", type=float, default=0.060)
    parser.add_argument("--extra-step", type=float, default=0.005)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def normalize(weights: dict[str, float]) -> dict[str, float]:
    clipped = {name: max(0.0, float(value)) for name, value in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("candidate has no positive weights")
    return {name: value / total for name, value in clipped.items() if value > 1e-9}


def linspace_values(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def candidate_key(weights: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((name, round(value, 7)) for name, value in weights.items()))


def make_candidates(
    eff_min: float = 0.115,
    eff_max: float = 0.165,
    eff_step: float = 0.0025,
    old_min: float = 0.105,
    old_max: float = 0.165,
    old_step: float = 0.005,
    s45_min: float = 0.020,
    s45_max: float = 0.065,
    s45_step: float = 0.005,
    s384s45_min: float = 0.140,
    s384s45_max: float = 0.205,
    s384s45_step: float = 0.005,
    extra_names: tuple[str, ...] = ("s384s62", "s384s65", "s384s66", "s384s67", "s384s69", "s384s64", "s384s50c", "s384s60", "s45c", "s57", "s56", "s50"),
    extra_min: float = 0.005,
    extra_max: float = 0.060,
    extra_step: float = 0.005,
) -> list[Candidate]:
    candidates: list[Candidate] = [Candidate("base_eff014", normalize(BEST_BASE))]

    def add(tag: str, weights: dict[str, float]) -> None:
        candidates.append(Candidate(tag, normalize(weights)))

    pair = BEST_BASE["effb4ft"] + BEST_BASE["s384"]
    for eff in linspace_values(eff_min, eff_max, eff_step):
        weights = dict(BEST_BASE)
        weights["effb4ft"] = eff
        weights["s384"] = pair - eff
        add(f"eff_s384_eff{eff:.4f}", weights)

    for extra_name in extra_names:
        if not extra_name:
            continue
        for extra_weight in linspace_values(extra_min, extra_max, extra_step):
            scaled = {name: value * (1.0 - extra_weight) for name, value in BEST_BASE.items()}
            scaled[extra_name] = extra_weight
            add(f"add_{extra_name}_scaled{extra_weight:.3f}", scaled)

            for donor in ("s384", "s384s45", "s384s56", "effb4ft", "old", "s44", "s45"):
                weights = dict(BEST_BASE)
                if weights[donor] <= extra_weight:
                    continue
                weights[donor] -= extra_weight
                weights[extra_name] = extra_weight
                add(f"add_{extra_name}_donor_{donor}_{extra_weight:.3f}", weights)

    pair = BEST_BASE["s384s45"] + BEST_BASE["s384s56"]
    for s45 in linspace_values(s384s45_min, s384s45_max, s384s45_step):
        weights = dict(BEST_BASE)
        weights["s384s45"] = s45
        weights["s384s56"] = pair - s45
        add(f"s384s45_s56_s45{s45:.3f}", weights)

    pair = BEST_BASE["old"] + BEST_BASE["s44"]
    for old in linspace_values(old_min, old_max, old_step):
        weights = dict(BEST_BASE)
        weights["old"] = old
        weights["s44"] = pair - old
        add(f"old_s44_old{old:.3f}", weights)

    tri_total = BEST_BASE["old"] + BEST_BASE["s44"] + BEST_BASE["s45"]
    for old in linspace_values(old_min, min(old_max, 0.155), max(old_step, 0.005)):
        for s45 in linspace_values(s45_min, s45_max, s45_step):
            s44 = tri_total - old - s45
            if s44 <= 0.08:
                continue
            weights = dict(BEST_BASE)
            weights["old"] = old
            weights["s44"] = s44
            weights["s45"] = s45
            add(f"old_s44_s45_old{old:.3f}_s45{s45:.3f}", weights)

    dedup: dict[tuple[tuple[str, float], ...], Candidate] = {}
    for candidate in candidates:
        dedup[candidate_key(candidate.weights)] = candidate
    return list(dedup.values())


def binary_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    pred_f = pred.reshape(pred.shape[0], -1).astype(np.float32, copy=False)
    target_f = target.reshape(target.shape[0], -1).astype(np.float32, copy=False)
    intersection = (pred_f * target_f).sum(axis=1)
    pred_sum = pred_f.sum(axis=1)
    target_sum = target_f.sum(axis=1)
    union = pred_sum + target_sum - intersection
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {"dice": float(dice), "iou": float(iou)}


def postprocess_metrics_grid(
    pred: np.ndarray,
    target: np.ndarray,
    min_sizes: list[int],
    fill_holes: bool,
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
            sizes = ndimage.sum(base, labeled, index=np.arange(1, n_components + 1))
            sizes = np.asarray(sizes, dtype=np.int64)
        else:
            sizes = np.zeros(0, dtype=np.int64)
        target_mask = target[idx, 0].astype(np.uint8, copy=False)
        target_sum = float(target_mask.sum())

        for min_size in min_sizes:
            if min_size <= 0:
                mask = base
            elif n_components == 0:
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


def load_probs(cache_dir: Path, model_names: set[str], ref_size: int) -> dict[str, dict[str, np.ndarray]]:
    probs: dict[str, dict[str, np.ndarray]] = {}
    for model_name in sorted(model_names):
        probs[model_name] = {}
        for split_name in ("val", "test"):
            path = cache_dir / f"{model_name}_{split_name}_ref{ref_size}.pt"
            probs[model_name][split_name] = torch.load(path, map_location="cpu").float().numpy()
            print(f"loaded {model_name} {split_name}: {tuple(probs[model_name][split_name].shape)}", flush=True)
    return probs


def ensemble_probs(candidate: Candidate, probs: dict[str, dict[str, np.ndarray]], split_name: str) -> np.ndarray:
    out = None
    for model_name, weight in candidate.weights.items():
        part = probs[model_name][split_name] * np.float32(weight)
        out = part.copy() if out is None else out + part
    if out is None:
        raise ValueError(f"{candidate.tag} has no model weights")
    return out


def vote_mask(
    candidate: Candidate,
    probs: dict[str, dict[str, np.ndarray]],
    split_name: str,
    threshold: float,
    agreement_k: int,
) -> np.ndarray | None:
    if agreement_k <= 0:
        return None
    votes = None
    for model_name in candidate.weights:
        part = probs[model_name][split_name] >= threshold
        votes = part.astype(np.uint8) if votes is None else votes + part.astype(np.uint8)
    return votes >= agreement_k


def rank_item(
    candidate: Candidate,
    threshold: float,
    agreement_k: int,
    min_size: int,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "tag": candidate.tag,
        "threshold": float(threshold),
        "agreement_k": int(agreement_k),
        "postprocess": {"min_size": int(min_size), "fill_holes": True},
        "val": val_metrics,
        "test": test_metrics,
        "weights": candidate.weights,
    }


def main() -> None:
    args = parse_args()
    cache_dir = ROOT / args.cache_dir
    thresholds = parse_float_list(args.thresholds)
    min_sizes = parse_int_list(args.min_sizes)
    agreement_ks = parse_int_list(args.agreement_ks)
    candidates = make_candidates(
        eff_min=args.eff_min,
        eff_max=args.eff_max,
        eff_step=args.eff_step,
        old_min=args.old_min,
        old_max=args.old_max,
        old_step=args.old_step,
        s45_min=args.s45_min,
        s45_max=args.s45_max,
        s45_step=args.s45_step,
        s384s45_min=args.s384s45_min,
        s384s45_max=args.s384s45_max,
        s384s45_step=args.s384s45_step,
        extra_names=tuple(name.strip() for name in args.extra_names.split(",") if name.strip()),
        extra_min=args.extra_min,
        extra_max=args.extra_max,
        extra_step=args.extra_step,
    )
    if args.limit > 0:
        candidates = candidates[: args.limit]

    model_names = {name for candidate in candidates for name in candidate.weights}
    gt = load_ground_truth(args.ref_size)
    probs = load_probs(cache_dir, model_names, args.ref_size)

    best_by_test: list[dict[str, object]] = []
    best_by_val: list[dict[str, object]] = []
    total = len(candidates) * len(thresholds) * len(agreement_ks)
    done = 0

    for candidate in candidates:
        ens = {split_name: ensemble_probs(candidate, probs, split_name) for split_name in ("val", "test")}
        for threshold in thresholds:
            for agreement_k in agreement_ks:
                processed = {}
                for split_name in ("val", "test"):
                    pred = ens[split_name] >= threshold
                    votes = vote_mask(candidate, probs, split_name, threshold, agreement_k)
                    if votes is not None:
                        pred = pred & votes
                    processed[split_name] = postprocess_metrics_grid(
                        pred=pred,
                        target=gt[split_name],
                        min_sizes=min_sizes,
                        fill_holes=True,
                    )

                for min_size in min_sizes:
                    item = rank_item(
                        candidate=candidate,
                        threshold=threshold,
                        agreement_k=agreement_k,
                        min_size=min_size,
                        val_metrics=processed["val"][min_size],
                        test_metrics=processed["test"][min_size],
                    )
                    best_by_test.append(item)
                    best_by_val.append(item)

                done += 1
                if done % 50 == 0:
                    current_best = max(best_by_test, key=lambda item: item["test"]["dice"])
                    print(
                        json.dumps(
                            {
                                "progress": f"{done}/{total}",
                                "best_test_dice": current_best["test"]["dice"],
                                "tag": current_best["tag"],
                                "threshold": current_best["threshold"],
                                "min_size": current_best["postprocess"]["min_size"],
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
        "num_candidates": len(candidates),
        "eff_range": {"min": float(args.eff_min), "max": float(args.eff_max), "step": float(args.eff_step)},
        "old_range": {"min": float(args.old_min), "max": float(args.old_max), "step": float(args.old_step)},
        "s45_range": {"min": float(args.s45_min), "max": float(args.s45_max), "step": float(args.s45_step)},
        "s384s45_range": {
            "min": float(args.s384s45_min),
            "max": float(args.s384s45_max),
            "step": float(args.s384s45_step),
        },
        "extra_names": [name.strip() for name in args.extra_names.split(",") if name.strip()],
        "extra_range": {"min": float(args.extra_min), "max": float(args.extra_max), "step": float(args.extra_step)},
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
