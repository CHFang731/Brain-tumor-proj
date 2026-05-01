#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
    parser = argparse.ArgumentParser(description="Search weighted TTA ensemble with optional post-processing.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="reports/ensemble_postprocess_search.json")
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


def postprocess_mask(mask: np.ndarray, min_size: int, keep_largest_only: bool, fill_holes: bool) -> np.ndarray:
    work = mask.astype(np.uint8)
    if fill_holes:
        work = ndimage.binary_fill_holes(work).astype(np.uint8)
    labeled, n = ndimage.label(work)
    if n == 0:
        return work

    sizes = ndimage.sum(work, labeled, index=np.arange(1, n + 1))
    sizes = np.asarray(sizes, dtype=np.int64)
    keep = np.zeros(n + 1, dtype=np.uint8)
    if keep_largest_only:
        keep[np.argmax(sizes) + 1] = 1
    else:
        for idx, size in enumerate(sizes, start=1):
            if int(size) >= min_size:
                keep[idx] = 1
    out = keep[labeled]
    return out.astype(np.uint8)


def binary_metrics_from_masks(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    preds_f = preds.flatten(1)
    targets_f = targets.flatten(1)
    tp = (preds_f * targets_f).sum(dim=1)
    fp = (preds_f * (1.0 - targets_f)).sum(dim=1)
    fn = ((1.0 - preds_f) * targets_f).sum(dim=1)
    tn = ((1.0 - preds_f) * (1.0 - targets_f)).sum(dim=1)
    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    acc = ((tp + tn + eps) / (tp + tn + fp + fn + eps)).mean().item()
    return {"dice": float(dice), "iou": float(iou), "pixel_accuracy": float(acc)}


def main() -> None:
    args = parse_args()

    specs = [
        ModelSpec(
            name="old_unet34_256",
            config="configs/segmentation_2d_smp_long.yaml",
            checkpoint="models/smp_unet_resnet34/best_unet2d.pt",
            image_size=256,
            in_channels=1,
        ),
        ModelSpec(
            name="deeplab50_256",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_256_focal.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_256_focal/best_unet2d.pt",
            image_size=256,
            in_channels=3,
        ),
        ModelSpec(
            name="deeplab50_320_seed42",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_finetune.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_320_finetune/best_unet2d.pt",
            image_size=320,
            in_channels=3,
        ),
        ModelSpec(
            name="deeplab50_320_seed43",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed43.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_320_seed43/best_unet2d.pt",
            image_size=320,
            in_channels=3,
        ),
        ModelSpec(
            name="deeplab50_320_seed44",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed44.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_320_seed44/best_unet2d.pt",
            image_size=320,
            in_channels=3,
        ),
    ]

    device = torch.device(args.device)
    ref_cfg = load_config(ROOT / specs[0].config)
    split = make_or_load_split(
        dataset_root=ROOT / ref_cfg["data"]["dataset_root"],
        split_json=ROOT / ref_cfg["data"]["split_json"],
        train_fraction=float(ref_cfg["split"].get("train_fraction", 0.7)),
        val_fraction=float(ref_cfg["split"].get("val_fraction", 0.15)),
        seed=int(ref_cfg["training"].get("seed", 42)),
    )
    val_items = split["val"]
    test_items = split["test"]

    gt_val_ds = BrainTumorSegDataset(val_items, image_size=256, training=False, in_channels=1)
    gt_test_ds = BrainTumorSegDataset(test_items, image_size=256, training=False, in_channels=1)
    gt_val = torch.stack([gt_val_ds[i]["mask"] for i in range(len(gt_val_ds))], dim=0).float()
    gt_test = torch.stack([gt_test_ds[i]["mask"] for i in range(len(gt_test_ds))], dim=0).float()

    cache: dict[str, dict[str, torch.Tensor]] = {}
    for spec in specs:
        ckpt_path = ROOT / spec.checkpoint
        if not ckpt_path.exists():
            continue
        cfg = load_config(ROOT / spec.config)
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        cache[spec.name] = {}
        for split_name, items in (("val", val_items), ("test", test_items)):
            ds = BrainTumorSegDataset(
                items,
                image_size=spec.image_size,
                training=False,
                in_channels=spec.in_channels,
            )
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
            probs = []
            with torch.no_grad():
                for batch in loader:
                    x = batch["image"].to(device, non_blocking=True)
                    p = tta_prob(model, x).cpu()
                    p = resize_to(p, (256, 256))
                    probs.append(p)
            cache[spec.name][split_name] = torch.cat(probs, dim=0)

    names = sorted(cache.keys())
    if len(names) < 2:
        raise RuntimeError("Need at least 2 available models for ensemble search.")

    combos: list[tuple[str, ...]] = []
    for k in (2, 3, 4):
        combos.extend(itertools.combinations(names, k))

    weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = [x / 100.0 for x in range(25, 51)]
    min_sizes = [0, 64, 128, 256, 512]
    fill_holes_grid = [False, True]
    keep_largest_grid = [False, True]

    best = None

    for combo in combos:
        k = len(combo)
        # Coarse simplex weights.
        if k == 2:
            weights_list = [[w, 1.0 - w] for w in weight_grid]
        elif k == 3:
            weights_list = []
            for w0 in weight_grid:
                for w1 in weight_grid:
                    w2 = 1.0 - w0 - w1
                    if w2 < -1e-9:
                        continue
                    if abs(round(w2, 1) - w2) > 1e-9:
                        continue
                    w2 = round(w2, 1)
                    if w2 < 0 or w2 > 1:
                        continue
                    weights_list.append([w0, w1, w2])
        else:
            weights_list = []
            for w0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for w1 in [0.1, 0.2, 0.3, 0.4]:
                    for w2 in [0.1, 0.2, 0.3]:
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 < -1e-9:
                            continue
                        w3 = round(w3, 1)
                        if w3 < 0 or w3 > 1:
                            continue
                        weights_list.append([w0, w1, w2, w3])

        probs_val = [cache[name]["val"] for name in combo]
        probs_test = [cache[name]["test"] for name in combo]

        for weights in weights_list:
            if abs(sum(weights) - 1.0) > 1e-6:
                continue
            p_val = sum(w * p for w, p in zip(weights, probs_val))
            p_test = sum(w * p for w, p in zip(weights, probs_test))

            for th in thresholds:
                raw_val = (p_val >= th).float()
                raw_test = (p_test >= th).float()

                for min_size in min_sizes:
                    for fill_holes in fill_holes_grid:
                        for keep_largest in keep_largest_grid:
                            if keep_largest and min_size > 0:
                                continue
                            pp_val = []
                            pp_test = []
                            for i in range(raw_val.shape[0]):
                                mv = postprocess_mask(
                                    raw_val[i, 0].numpy(),
                                    min_size=min_size,
                                    keep_largest_only=keep_largest,
                                    fill_holes=fill_holes,
                                )
                                mt = postprocess_mask(
                                    raw_test[i, 0].numpy(),
                                    min_size=min_size,
                                    keep_largest_only=keep_largest,
                                    fill_holes=fill_holes,
                                )
                                pp_val.append(torch.from_numpy(mv).unsqueeze(0).float())
                                pp_test.append(torch.from_numpy(mt).unsqueeze(0).float())
                            pred_val = torch.stack(pp_val, dim=0)
                            pred_test = torch.stack(pp_test, dim=0)

                            m_val = binary_metrics_from_masks(pred_val, gt_val)
                            if best is None or m_val["dice"] > best["val_dice"]:
                                m_test = binary_metrics_from_masks(pred_test, gt_test)
                                best = {
                                    "models": list(combo),
                                    "weights": [float(x) for x in weights],
                                    "threshold": float(th),
                                    "postprocess": {
                                        "min_size": int(min_size),
                                        "fill_holes": bool(fill_holes),
                                        "keep_largest_only": bool(keep_largest),
                                    },
                                    "val_dice": m_val["dice"],
                                    "val_iou": m_val["iou"],
                                    "val_pixel_accuracy": m_val["pixel_accuracy"],
                                    "test_dice": m_test["dice"],
                                    "test_iou": m_test["iou"],
                                    "test_pixel_accuracy": m_test["pixel_accuracy"],
                                }
                                print(json.dumps(best, ensure_ascii=True))

    if best is None:
        raise RuntimeError("No valid ensemble setting found.")

    out_path = ROOT / args.output
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(json.dumps(best, ensure_ascii=True))


if __name__ == "__main__":
    main()
