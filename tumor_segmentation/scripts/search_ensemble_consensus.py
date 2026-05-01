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
    parser = argparse.ArgumentParser(
        description="Search consensus/uncertainty post-processing for a fixed weighted TTA ensemble."
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="reports/ensemble_consensus_search_20260501.json")
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


def apply_rules(
    probs: torch.Tensor,
    probs_by_model: list[torch.Tensor],
    threshold: float,
    agreement_k: int,
    std_max: float,
) -> torch.Tensor:
    pred = probs >= threshold
    if agreement_k > 0:
        votes = torch.zeros_like(probs, dtype=torch.int32)
        for p in probs_by_model:
            votes += (p >= threshold).int()
        pred = pred & (votes >= agreement_k)
    if std_max < 0.999:
        stack = torch.stack(probs_by_model, dim=0)
        std = stack.std(dim=0, unbiased=False)
        pred = pred & (std <= std_max)
    return pred.float()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Fixed 4-model pool and near-best weights from prior microgrid search.
    specs = [
        ModelSpec(
            name="old",
            config="configs/segmentation_2d_smp_long.yaml",
            checkpoint="models/smp_unet_resnet34/best_unet2d.pt",
            image_size=256,
            in_channels=1,
        ),
        ModelSpec(
            name="s44",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed44.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_320_seed44/best_unet2d.pt",
            image_size=320,
            in_channels=3,
        ),
        ModelSpec(
            name="s45",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed45.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_320_seed45/best_unet2d.pt",
            image_size=320,
            in_channels=3,
        ),
        ModelSpec(
            name="s51",
            config="configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed51_plateau_conservative.yaml",
            checkpoint="models/smp_deeplabv3plus_resnet50_320_seed51_plateau_conservative/best_unet2d.pt",
            image_size=320,
            in_channels=3,
        ),
    ]
    weights = {"old": 0.2024, "s44": 0.4288, "s45": 0.2088, "s51": 0.16}

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
        cfg = load_config(ROOT / spec.config)
        model = build_model(cfg).to(device)
        state = torch.load(ROOT / spec.checkpoint, map_location=device)
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
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            probs = []
            with torch.no_grad():
                for batch in loader:
                    x = batch["image"].to(device, non_blocking=True)
                    p = tta_prob(model, x).cpu()
                    probs.append(resize_to(p, (256, 256)))
            cache[spec.name][split_name] = torch.cat(probs, dim=0)

    names = [s.name for s in specs]
    probs_val_by_model = [cache[n]["val"] for n in names]
    probs_test_by_model = [cache[n]["test"] for n in names]
    probs_val = sum(weights[n] * cache[n]["val"] for n in names)
    probs_test = sum(weights[n] * cache[n]["test"] for n in names)

    thresholds = [round(x / 100.0, 2) for x in range(35, 43)]
    agreement_ks = [0, 2, 3]
    std_maxes = [1.0, 0.20, 0.15, 0.12]
    min_sizes = [0, 64, 96, 128]
    fill_holes_grid = [False, True]

    best = None
    for th in thresholds:
        for k in agreement_ks:
            for std_max in std_maxes:
                if k == 0 and std_max < 0.999:
                    # Keep some search efficiency; uncertainty filter is most meaningful with consensus.
                    continue

                raw_val = apply_rules(
                    probs=probs_val,
                    probs_by_model=probs_val_by_model,
                    threshold=th,
                    agreement_k=k,
                    std_max=std_max,
                )
                raw_test = apply_rules(
                    probs=probs_test,
                    probs_by_model=probs_test_by_model,
                    threshold=th,
                    agreement_k=k,
                    std_max=std_max,
                )

                for min_size in min_sizes:
                    for fill_holes in fill_holes_grid:
                        pp_val = []
                        pp_test = []
                        for i in range(raw_val.shape[0]):
                            mv = postprocess_mask(raw_val[i, 0].numpy(), min_size=min_size, fill_holes=fill_holes)
                            pp_val.append(torch.from_numpy(mv).unsqueeze(0).float())
                        for i in range(raw_test.shape[0]):
                            mt = postprocess_mask(raw_test[i, 0].numpy(), min_size=min_size, fill_holes=fill_holes)
                            pp_test.append(torch.from_numpy(mt).unsqueeze(0).float())
                        pred_val = torch.stack(pp_val, dim=0)
                        pred_test = torch.stack(pp_test, dim=0)

                        m_val = binary_metrics_from_masks(pred_val, gt_val)
                        if best is None or m_val["dice"] > best["val_dice"]:
                            m_test = binary_metrics_from_masks(pred_test, gt_test)
                            best = {
                                "weights": {k1: float(v1) for k1, v1 in weights.items()},
                                "threshold": float(th),
                                "agreement_k": int(k),
                                "std_max": float(std_max),
                                "postprocess": {
                                    "min_size": int(min_size),
                                    "fill_holes": bool(fill_holes),
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
        raise RuntimeError("No valid setting found.")

    out_path = ROOT / args.output
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    print(json.dumps(best, ensure_ascii=True))


if __name__ == "__main__":
    main()
