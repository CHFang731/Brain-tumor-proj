#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one cached ensemble candidate in full precision.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--weights-json", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--agreement-k", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    parser.add_argument("--smooth-votes", action="store_true")
    return parser.parse_args()


def postprocess_mask(mask, min_size: int, fill_holes: bool):
    work = mask.astype("uint8")
    if fill_holes:
        work = ndimage.binary_fill_holes(work).astype("uint8")
    if min_size <= 0:
        return work
    labeled, n = ndimage.label(work)
    if n == 0:
        return work
    sizes = ndimage.sum(work, labeled, index=range(1, n + 1))
    keep = torch.zeros(n + 1, dtype=torch.uint8).numpy()
    for idx, size in enumerate(sizes, start=1):
        if int(size) >= min_size:
            keep[idx] = 1
    return keep[labeled].astype("uint8")


def metrics(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    preds_f = preds.flatten(1).float()
    targets_f = targets.flatten(1).float()
    intersection = (preds_f * targets_f).sum(dim=1)
    pred_sum = preds_f.sum(dim=1)
    target_sum = targets_f.sum(dim=1)
    union = pred_sum + target_sum - intersection
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item()
    iou = ((intersection + eps) / (union + eps)).mean().item()
    return {"dice": float(dice), "iou": float(iou)}


def main() -> None:
    args = parse_args()
    weights = json.loads(args.weights_json)
    cache_dir = ROOT / args.cache_dir

    ref_cfg = load_config(ROOT / "configs/segmentation_2d_smp_long.yaml")
    split = make_or_load_split(
        dataset_root=ROOT / ref_cfg["data"]["dataset_root"],
        split_json=ROOT / ref_cfg["data"]["split_json"],
        train_fraction=float(ref_cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(ref_cfg["split"].get("val_fraction", 0.15)),
        seed=int(ref_cfg["training"].get("seed", 42)),
    )

    result = {}
    for split_name in ("val", "test"):
        gt_ds = BrainTumorSegDataset(split[split_name], image_size=args.ref_size, training=False, in_channels=1)
        gt = torch.stack([gt_ds[i]["mask"] for i in range(len(gt_ds))], dim=0).float()
        probs_by_model = {
            name: torch.load(cache_dir / f"{name}_{split_name}_ref{args.ref_size}.pt", map_location="cpu").float()
            for name in weights
        }
        if args.smooth_sigma > 0 and args.smooth_votes:
            probs_by_model = {
                name: torch.from_numpy(
                    ndimage.gaussian_filter(
                        prob.numpy(),
                        sigma=(0, 0, float(args.smooth_sigma), float(args.smooth_sigma)),
                    )
                ).float()
                for name, prob in probs_by_model.items()
            }
        probs = sum(float(weight) * probs_by_model[name] for name, weight in weights.items())
        if args.smooth_sigma > 0 and not args.smooth_votes:
            smoothed = ndimage.gaussian_filter(
                probs.numpy(),
                sigma=(0, 0, float(args.smooth_sigma), float(args.smooth_sigma)),
            )
            probs = torch.from_numpy(smoothed).float()
        pred = probs >= float(args.threshold)
        if int(args.agreement_k) > 0:
            votes = torch.zeros_like(probs, dtype=torch.int16)
            for p in probs_by_model.values():
                votes += (p >= float(args.threshold)).short()
            pred = pred & (votes >= int(args.agreement_k))

        pred_f = pred.float()
        if args.min_size > 0 or args.fill_holes:
            masks = []
            for idx in range(pred_f.shape[0]):
                mask = postprocess_mask(pred_f[idx, 0].numpy(), min_size=args.min_size, fill_holes=args.fill_holes)
                masks.append(torch.from_numpy(mask).unsqueeze(0).float())
            pred_f = torch.stack(masks, dim=0)

        result[split_name] = metrics(pred_f, gt)

    result.update(
        {
            "weights": weights,
            "threshold": float(args.threshold),
            "agreement_k": int(args.agreement_k),
            "postprocess": {"min_size": int(args.min_size), "fill_holes": bool(args.fill_holes)},
            "smooth_sigma": float(args.smooth_sigma),
            "smooth_votes": bool(args.smooth_votes),
        }
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
