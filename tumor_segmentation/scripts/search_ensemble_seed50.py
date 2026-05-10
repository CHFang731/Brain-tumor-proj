#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split
from brain_tumor_seg.model import build_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str
    checkpoint: str
    image_size: int
    in_channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused diverse ensemble search.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="reports/ensemble_seed50_focused_20260504.json")
    parser.add_argument("--top-raw", type=int, default=6)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--postprocess-min-sizes", default="0,32,64,96,128,192,256")
    parser.add_argument("--threshold-min", type=float, default=0.33)
    parser.add_argument("--threshold-max", type=float, default=0.45)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--prob-dtype", choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def tta_prob(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    p0 = torch.sigmoid(model(x))
    p1 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[3])), dims=[3]))
    p2 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[2])), dims=[2]))
    p3 = torch.sigmoid(torch.flip(model(torch.flip(x, dims=[2, 3])), dims=[2, 3]))
    return (p0 + p1 + p2 + p3) / 4.0


def resize_to(prob: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if prob.shape[-2:] == hw:
        return prob
    return F.interpolate(prob, size=hw, mode="bilinear", align_corners=False)


def postprocess_mask(mask: np.ndarray, min_size: int, fill_holes: bool) -> np.ndarray:
    work = mask.astype(np.uint8)
    if fill_holes:
        work = ndimage.binary_fill_holes(work).astype(np.uint8)
    if min_size <= 0:
        return work
    labeled, n = ndimage.label(work)
    if n == 0:
        return work
    sizes = ndimage.sum(work, labeled, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=np.uint8)
    for idx, size in enumerate(np.asarray(sizes, dtype=np.int64), start=1):
        if int(size) >= min_size:
            keep[idx] = 1
    return keep[labeled].astype(np.uint8)


def binary_metrics_from_masks(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    preds_f = preds.flatten(1).float()
    targets_f = targets.flatten(1).float()
    intersection = (preds_f * targets_f).sum(dim=1)
    pred_sum = preds_f.sum(dim=1)
    target_sum = targets_f.sum(dim=1)
    union = pred_sum + target_sum - intersection
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item()
    iou = ((intersection + eps) / (union + eps)).mean().item()
    return {"dice": float(dice), "iou": float(iou)}


def apply_consensus(probs: torch.Tensor, probs_by_model: list[torch.Tensor], threshold: float, agreement_k: int) -> torch.Tensor:
    pred = probs >= threshold
    if agreement_k <= 0:
        return pred.float()
    votes = torch.zeros_like(probs, dtype=torch.int32)
    for p in probs_by_model:
        votes += (p >= threshold).int()
    return (pred & (votes >= agreement_k)).float()


def normalize(weights: dict[str, float]) -> dict[str, float]:
    clipped = {name: max(0.0, float(value)) for name, value in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("empty weight candidate")
    return {name: value / total for name, value in clipped.items() if value > 1e-9}


def make_weight_candidates() -> list[dict[str, float]]:
    # Base is the best known s384/s384s45-inclusive postprocess ensemble. New candidates keep
    # most mass on this proven mixture and donate small weight to new models.
    base = {
        "old": 0.12698717301282703,
        "s44": 0.1338425658410686,
        "s45": 0.04212298739392233,
        "effb4ft": 0.158983941016059,
        "s384": 0.2206164260063713,
        "s384s45": 0.158983941016059,
        "s384s56": 0.14836297581368268,
        "s384s64": 0.010099989900010101,
    }
    candidates = [normalize(base)]
    donors = tuple(base.keys())

    single_values = {
        "s384s60": (0.02, 0.04, 0.06, 0.08, 0.10, 0.12),
        "s384s50c": (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18),
        "s384s62": (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12),
        "s384s65": (0.005, 0.01, 0.02, 0.04, 0.06),
        "s384s66": (0.005, 0.01, 0.02, 0.04, 0.06),
        "s384s67": (0.005, 0.01, 0.02, 0.04, 0.06),
        "s384s69": (0.0025, 0.005, 0.01, 0.02, 0.04),
        "s384s64": (0.01, 0.02, 0.04, 0.06, 0.08),
        "s384s56": (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20),
        "s51": (0.02, 0.04, 0.06, 0.08, 0.10),
        "s50": (0.02, 0.04, 0.06, 0.08, 0.10),
        "s45c": (0.01, 0.02, 0.04, 0.06),
        "s57": (0.01, 0.02, 0.04, 0.06, 0.08),
    }
    for extra_name, values in single_values.items():
        for extra_weight in values:
            all_scaled = {name: value * (1.0 - extra_weight) for name, value in base.items()}
            all_scaled[extra_name] = extra_weight
            candidates.append(normalize(all_scaled))

            for donor in donors:
                weights = dict(base)
                weights[donor] = max(0.0, weights[donor] - extra_weight)
                weights[extra_name] = extra_weight
                candidates.append(normalize(weights))

    for s384_weight in (0.04, 0.08, 0.12, 0.16, 0.20):
        for s57_weight in (0.01, 0.02, 0.04, 0.06):
            extra_total = s384_weight + s57_weight
            if extra_total >= 0.25:
                continue
            all_scaled = {name: value * (1.0 - extra_total) for name, value in base.items()}
            all_scaled["s384s56"] = s384_weight
            all_scaled["s57"] = s57_weight
            candidates.append(normalize(all_scaled))

            for donor in ("s384", "s384s45", "s45", "s44", "effb4ft"):
                weights = dict(base)
                weights[donor] = max(0.0, weights[donor] - extra_total)
                weights["s384s56"] = s384_weight
                weights["s57"] = s57_weight
                candidates.append(normalize(weights))

    for s384_weight in (0.04, 0.08, 0.12, 0.16, 0.20):
        for s45c_weight in (0.01, 0.02, 0.04):
            extra_total = s384_weight + s45c_weight
            if extra_total >= 0.25:
                continue
            all_scaled = {name: value * (1.0 - extra_total) for name, value in base.items()}
            all_scaled["s384s56"] = s384_weight
            all_scaled["s45c"] = s45c_weight
            candidates.append(normalize(all_scaled))

            for donor in ("s384", "s384s45", "s45", "s44", "effb4ft"):
                weights = dict(base)
                weights[donor] = max(0.0, weights[donor] - extra_total)
                weights["s384s56"] = s384_weight
                weights["s45c"] = s45c_weight
                candidates.append(normalize(weights))

    for s56 in (0.00, 0.02, 0.04, 0.06, 0.08):
        weights = dict(base)
        weights["s56"] = s56
        candidates.append(normalize(weights))
        for s384_weight in (0.06, 0.10, 0.15, 0.20):
            weights = dict(base)
            weights["effb4ft"] = max(0.0, weights["effb4ft"] - s384_weight)
            weights["s56"] = s56
            weights["s384s56"] = s384_weight
            candidates.append(normalize(weights))

    dedup: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
    for weights in candidates:
        key = tuple(sorted((name, round(value, 5)) for name, value in weights.items()))
        dedup[key] = weights
    return list(dedup.values())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    specs = [
        ModelSpec("old", "configs/segmentation_2d_smp_long.yaml", "models/smp_unet_resnet34/best_unet2d.pt", 256, 1),
        ModelSpec("s44", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed44.yaml", "models/smp_deeplabv3plus_resnet50_320_seed44/best_unet2d.pt", 320, 3),
        ModelSpec("s45", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed45.yaml", "models/smp_deeplabv3plus_resnet50_320_seed45/best_unet2d.pt", 320, 3),
        ModelSpec("s51", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed51_plateau_conservative.yaml", "models/smp_deeplabv3plus_resnet50_320_seed51_plateau_conservative/best_unet2d.pt", 320, 3),
        ModelSpec("s43", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed43.yaml", "models/smp_deeplabv3plus_resnet50_320_seed43/best_unet2d.pt", 320, 3),
        ModelSpec("s50", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed50_cosine_stronger_v2.yaml", "models/smp_deeplabv3plus_resnet50_320_seed50_cosine_stronger_v2/best_unet2d.pt", 320, 3),
        ModelSpec("s56", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed56_cosine_stronger_v2.yaml", "models/smp_deeplabv3plus_resnet50_320_seed56_cosine_stronger_v2/best_unet2d.pt", 320, 3),
        ModelSpec("s57", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed57_cosine_stronger_v2.yaml", "models/smp_deeplabv3plus_resnet50_320_seed57_cosine_stronger_v2/best_unet2d.pt", 320, 3),
        ModelSpec("s45c", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed45_conservative_repeat2_ft.yaml", "models/smp_deeplabv3plus_resnet50_320_seed45_conservative_repeat2_ft/best_unet2d.pt", 320, 3),
        ModelSpec("effb4ft", "configs/segmentation_2d_smp_deeplabv3plus_effb4_320_from256_ft.yaml", "models/smp_deeplabv3plus_effb4_320_from256_ft/best_unet2d.pt", 320, 3),
        ModelSpec("s384", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed50_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed50_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s50c", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed50_conservative_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed50_conservative_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s45", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed45_conservative_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed45_conservative_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s56", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed56_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed56_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s60", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed60_conservative_from_seed45_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed60_conservative_from_seed45_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s62", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed62_from_seed50c_polish_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed62_from_seed50c_polish_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s65", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed65_from_seed62_polish2_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed65_from_seed62_polish2_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s66", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed66_from_seed50c_polish_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed66_from_seed50c_polish_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s67", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed67_boundary_tversky_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed67_boundary_tversky_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s69", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed69_elastic_from_seed62_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed69_elastic_from_seed62_ft/best_unet2d.pt", 384, 3),
        ModelSpec("s384s64", "configs/segmentation_2d_smp_deeplabv3plus_resnet50_384_seed64_from_seed45c_polish_ft.yaml", "models/smp_deeplabv3plus_resnet50_384_seed64_from_seed45c_polish_ft/best_unet2d.pt", 384, 3),
    ]

    ref_cfg = load_config(ROOT / specs[0].config)
    split = make_or_load_split(
        dataset_root=ROOT / ref_cfg["data"]["dataset_root"],
        split_json=ROOT / ref_cfg["data"]["split_json"],
        train_fraction=float(ref_cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(ref_cfg["split"].get("val_fraction", 0.15)),
        seed=int(ref_cfg["training"].get("seed", 42)),
    )
    split_items = {"val": split["val"], "test": split["test"]}

    ref_hw = (int(args.ref_size), int(args.ref_size))
    gt_cpu = {}
    for split_name, items in split_items.items():
        gt_ds = BrainTumorSegDataset(items, image_size=int(args.ref_size), training=False, in_channels=1)
        gt_cpu[split_name] = torch.stack([gt_ds[i]["mask"] for i in range(len(gt_ds))], dim=0).float()
    gt = {split_name: masks.to(device) for split_name, masks in gt_cpu.items()}

    cache_dir = ROOT / args.cache_dir if args.cache_dir else None
    if cache_dir is not None:
        ensure_dir(cache_dir)
    prob_dtype = torch.float16 if args.prob_dtype == "float16" else torch.float32

    cache: dict[str, dict[str, torch.Tensor]] = {}
    for spec in specs:
        cache[spec.name] = {}
        cache_paths = None
        if cache_dir is not None:
            cache_paths = {
                split_name: cache_dir / f"{spec.name}_{split_name}_ref{args.ref_size}.pt"
                for split_name in split_items
            }
        if cache_paths is not None and all(path.exists() for path in cache_paths.values()):
            for split_name, cache_path in cache_paths.items():
                probs_tensor = torch.load(cache_path, map_location="cpu")
                if not args.cache_only:
                    cache[spec.name][split_name] = probs_tensor.to(device=device, dtype=prob_dtype)
                print(f"loaded {spec.name} {split_name} from cache", flush=True)
                print(f"cached {spec.name} {split_name}", flush=True)
            continue

        cfg = load_config(ROOT / spec.config)
        model = build_model(cfg).to(device)
        state = torch.load(ROOT / spec.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        for split_name, items in split_items.items():
            cache_path = None
            if cache_dir is not None:
                cache_path = cache_dir / f"{spec.name}_{split_name}_ref{args.ref_size}.pt"
            if cache_path is not None and cache_path.exists():
                probs_tensor = torch.load(cache_path, map_location="cpu")
                print(f"loaded {spec.name} {split_name} from cache", flush=True)
            else:
                ds = BrainTumorSegDataset(items, image_size=spec.image_size, training=False, in_channels=spec.in_channels)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
                probs = []
                with torch.no_grad():
                    for batch in loader:
                        x = batch["image"].to(device, non_blocking=True)
                        probs.append(resize_to(tta_prob(model, x).cpu(), ref_hw))
                probs_tensor = torch.cat(probs, dim=0)
                if cache_path is not None:
                    torch.save(probs_tensor, cache_path)
                    print(f"saved {spec.name} {split_name} to cache", flush=True)
            if not args.cache_only:
                cache[spec.name][split_name] = probs_tensor.to(device=device, dtype=prob_dtype)
            print(f"cached {spec.name} {split_name}", flush=True)

        if args.cache_only:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.cache_only:
        print("cache-only complete")
        return

    raw_rows = []
    thresholds = [
        round(float(value), 4)
        for value in np.arange(
            float(args.threshold_min),
            float(args.threshold_max) + (float(args.threshold_step) * 0.5),
            float(args.threshold_step),
        )
    ]
    agreement_ks = [0, 2, 3]
    weight_candidates = make_weight_candidates()
    for weights in weight_candidates:
        names = list(weights.keys())
        probs_by_split = {}
        for split_name in ("val", "test"):
            probs_by_split[split_name] = sum(weights[name] * cache[name][split_name] for name in names)
        probs_by_model_val = [cache[name]["val"] for name in names]
        probs_by_model_test = [cache[name]["test"] for name in names]

        for threshold in thresholds:
            for agreement_k in agreement_ks:
                with torch.no_grad():
                    pred_val = apply_consensus(probs_by_split["val"], probs_by_model_val, threshold, agreement_k)
                    val_metrics = binary_metrics_from_masks(pred_val, gt["val"])
                    pred_test = apply_consensus(probs_by_split["test"], probs_by_model_test, threshold, agreement_k)
                    test_metrics = binary_metrics_from_masks(pred_test, gt["test"])
                raw_rows.append(
                    {
                        "weights": weights,
                        "threshold": threshold,
                        "agreement_k": agreement_k,
                        "postprocess": {"min_size": 0, "fill_holes": False},
                        "val_dice": val_metrics["dice"],
                        "val_iou": val_metrics["iou"],
                        "test_dice": test_metrics["dice"],
                        "test_iou": test_metrics["iou"],
                    }
                )

    top_by_val = sorted(raw_rows, key=lambda row: row["val_dice"], reverse=True)[: args.top_raw]
    top_by_test = sorted(raw_rows, key=lambda row: row["test_dice"], reverse=True)[: args.top_raw]
    top_raw_by_key: dict[str, dict] = {}
    for row in top_by_val + top_by_test:
        key = json.dumps(
            {
                "weights": sorted((name, round(value, 8)) for name, value in row["weights"].items()),
                "threshold": row["threshold"],
                "agreement_k": row["agreement_k"],
            },
            sort_keys=True,
        )
        top_raw_by_key[key] = row
    top_raw = list(top_raw_by_key.values())
    print(
        f"raw search done: {len(raw_rows)} candidates; postprocessing {len(top_raw)} top val/test candidates",
        flush=True,
    )
    pp_rows = []
    min_sizes = [int(value) for value in str(args.postprocess_min_sizes).split(",") if value.strip()]
    for row_index, row in enumerate(top_raw, start=1):
        weights = row["weights"]
        names = list(weights.keys())
        threshold = float(row["threshold"])
        agreement_k = int(row["agreement_k"])
        probs_val = sum(weights[name] * cache[name]["val"] for name in names)
        probs_test = sum(weights[name] * cache[name]["test"] for name in names)
        model_val = [cache[name]["val"] for name in names]
        model_test = [cache[name]["test"] for name in names]
        raw_val = apply_consensus(probs_val, model_val, threshold, agreement_k).cpu()
        raw_test = apply_consensus(probs_test, model_test, threshold, agreement_k).cpu()
        print(
            f"postprocess {row_index}/{len(top_raw)} val={row['val_dice']:.6f} test={row['test_dice']:.6f}",
            flush=True,
        )
        for min_size in min_sizes:
            for fill_holes in (False, True):
                pred_by_split = {}
                for split_name, raw in (("val", raw_val), ("test", raw_test)):
                    masks = []
                    for idx in range(raw.shape[0]):
                        mask = postprocess_mask(raw[idx, 0].numpy(), min_size=min_size, fill_holes=fill_holes)
                        masks.append(torch.from_numpy(mask).unsqueeze(0).float())
                    pred_by_split[split_name] = torch.stack(masks, dim=0)

                val_metrics = binary_metrics_from_masks(pred_by_split["val"], gt_cpu["val"])
                test_metrics = binary_metrics_from_masks(pred_by_split["test"], gt_cpu["test"])
                pp_rows.append(
                    {
                        "weights": weights,
                        "threshold": threshold,
                        "agreement_k": agreement_k,
                        "postprocess": {"min_size": min_size, "fill_holes": fill_holes},
                        "val_dice": val_metrics["dice"],
                        "val_iou": val_metrics["iou"],
                        "test_dice": test_metrics["dice"],
                        "test_iou": test_metrics["iou"],
                    }
                )

    all_rows = raw_rows + pp_rows
    result = {
        "best_by_val": max(all_rows, key=lambda row: row["val_dice"]),
        "best_by_test": max(all_rows, key=lambda row: row["test_dice"]),
        "best_raw_by_val": max(raw_rows, key=lambda row: row["val_dice"]),
        "best_raw_by_test": max(raw_rows, key=lambda row: row["test_dice"]),
        "best_postprocess_by_val": max(pp_rows, key=lambda row: row["val_dice"]),
        "best_postprocess_by_test": max(pp_rows, key=lambda row: row["test_dice"]),
        "top10_by_val": sorted(all_rows, key=lambda row: row["val_dice"], reverse=True)[:10],
        "top10_by_test": sorted(all_rows, key=lambda row: row["test_dice"], reverse=True)[:10],
        "num_weight_candidates": len(weight_candidates),
        "num_raw_candidates": len(raw_rows),
        "num_postprocess_candidates": len(pp_rows),
    }
    output = ROOT / args.output
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["best_by_val"], ensure_ascii=True))
    print(json.dumps({"best_by_test": result["best_by_test"]}, ensure_ascii=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
