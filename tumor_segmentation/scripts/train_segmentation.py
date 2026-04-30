#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_seg.config import ensure_dir, load_config
from brain_tumor_seg.data import BrainTumorSegDataset, make_or_load_split
from brain_tumor_seg.metrics import binary_metrics, combined_loss
from brain_tumor_seg.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary U-Net on brain tumor segmentation dataset.")
    parser.add_argument("--config", default="configs/segmentation_2d.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Optional checkpoint path (relative to tumor_segmentation/) to continue training from.",
    )
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help="When resuming, ignore existing training_history.json and start a fresh history list.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    bce_weight: float,
    pos_weight,
    focal_weight: float,
    focal_gamma: float,
):
    import torch

    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(images)
            loss = combined_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                pos_weight=pos_weight,
                focal_weight=focal_weight,
                focal_gamma=focal_gamma,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics = binary_metrics(logits.detach(), masks)
        total_loss += float(loss.detach().cpu())
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_acc += metrics["pixel_accuracy"]

    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "pixel_accuracy": total_acc / n,
    }


def evaluate(model, loader, device, bce_weight: float, focal_weight: float, focal_gamma: float):
    import torch

    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            loss = combined_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                pos_weight=None,
                focal_weight=focal_weight,
                focal_gamma=focal_gamma,
            )
            metrics = binary_metrics(logits, masks)
            total_loss += float(loss.detach().cpu())
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            total_acc += metrics["pixel_accuracy"]

    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "pixel_accuracy": total_acc / n,
    }


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    config = load_config(ROOT / args.config)
    data_cfg = config["data"]
    train_cfg = config["training"]

    set_seed(int(train_cfg.get("seed", 42)))

    split = make_or_load_split(
        dataset_root=ROOT / data_cfg["dataset_root"],
        split_json=ROOT / data_cfg["split_json"],
        train_fraction=float(config["split"].get("train_fraction", 0.7)),
        val_fraction=float(config["split"].get("val_fraction", 0.15)),
        seed=int(train_cfg.get("seed", 42)),
    )

    image_size = int(data_cfg.get("image_size", 256))
    in_channels = int(config.get("model", {}).get("in_channels", 1))
    stronger_aug = bool(train_cfg.get("stronger_aug", False))
    train_ds = BrainTumorSegDataset(
        split["train"],
        image_size=image_size,
        training=True,
        in_channels=in_channels,
        stronger_aug=stronger_aug,
    )
    val_ds = BrainTumorSegDataset(
        split["val"],
        image_size=image_size,
        training=False,
        in_channels=in_channels,
        stronger_aug=False,
    )

    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(train_cfg.get("batch_size", 16))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = resolve_device(args.device)
    model = build_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    scheduler = None
    scheduler_type = str(train_cfg.get("scheduler", "none")).lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(train_cfg.get("scheduler_tmax", 30))), eta_min=float(train_cfg.get("scheduler_eta_min", 1e-6))
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 4)),
            min_lr=float(train_cfg.get("scheduler_min_lr", 1e-6)),
        )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pos_weight_value = float(train_cfg.get("pos_weight", 1.0))
    pos_weight = torch.tensor([pos_weight_value], device=device)
    bce_weight = float(train_cfg.get("bce_weight", 0.5))
    focal_weight = float(train_cfg.get("focal_weight", 0.0))
    focal_gamma = float(train_cfg.get("focal_gamma", 2.0))

    output_dir = ensure_dir(ROOT / data_cfg["output_dir"])
    ensure_dir(ROOT / "reports")

    best_dice = -1.0
    best_ckpt = output_dir / "best_unet2d.pt"
    history_path = output_dir / "training_history.json"
    history: list[dict[str, float | int]] = []
    patience = int(train_cfg.get("early_stopping_patience", 8))
    stale_epochs = 0
    start_epoch = 1
    resumed = False

    if args.resume_checkpoint:
        resume_path = ROOT / args.resume_checkpoint
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        ckpt_epoch = int(ckpt.get("epoch", 0))
        start_epoch = ckpt_epoch + 1
        best_dice = float(ckpt.get("best_val_dice", ckpt.get("val_metrics", {}).get("dice", -1.0)))
        resumed = True

        if history_path.exists() and not args.reset_history:
            with history_path.open("r", encoding="utf-8") as handle:
                loaded_history = json.load(handle)
            if isinstance(loaded_history, list):
                history = loaded_history

    start = time.time()
    run_epochs = int(train_cfg.get("epochs", 30))
    end_epoch = start_epoch + run_epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            bce_weight=bce_weight,
            pos_weight=pos_weight,
            focal_weight=focal_weight,
            focal_gamma=focal_gamma,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            bce_weight=bce_weight,
            focal_weight=focal_weight,
            focal_gamma=focal_gamma,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_val_dice": best_dice,
                },
                best_ckpt,
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "best_val_dice": best_dice,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=True))

        # Keep an up-to-date history file so progress is not lost on interruption.
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["dice"])
            else:
                scheduler.step()

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - start
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_val_dice": best_dice,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(best_ckpt),
        "resumed": resumed,
        "start_epoch": start_epoch,
        "end_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "train_count": len(split["train"]),
        "val_count": len(split["val"]),
        "test_count": len(split["test"]),
    }
    summary_path = ROOT / "reports" / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
