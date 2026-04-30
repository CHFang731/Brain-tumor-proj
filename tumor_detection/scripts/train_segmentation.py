#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import configure_runtime, ensure_dir, load_config
from brain_tumor.data.brats import make_or_load_split
from brain_tumor.data.transforms import segmentation_transforms
from brain_tumor.models.segmentation import create_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MONAI BraTS segmentation model.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    return parser.parse_args()


def main() -> None:
    configure_runtime(ROOT)
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete, Compose
    from monai.utils import set_determinism
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    config = load_config(parse_args().config)
    data_config = config["data"]
    train_config = config["training"]

    set_determinism(seed=int(train_config.get("seed", 42)))
    output_dir = ensure_dir(data_config["output_dir"])
    split = make_or_load_split(
        data_config["brats_root"],
        data_config["split_json"],
        seed=int(train_config.get("seed", 42)),
    )

    roi_size = tuple(int(v) for v in train_config["roi_size"])
    train_transform = segmentation_transforms(roi_size, training=True)
    val_transform = segmentation_transforms(roi_size, training=False)
    cache_rate = float(data_config.get("cache_rate", 0.0))
    dataset_cls = CacheDataset if cache_rate > 0 else Dataset
    train_ds = dataset_cls(split["train"], transform=train_transform, cache_rate=cache_rate) if cache_rate > 0 else dataset_cls(split["train"], transform=train_transform)
    val_ds = dataset_cls(split["val"], transform=val_transform, cache_rate=cache_rate) if cache_rate > 0 else dataset_cls(split["val"], transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 4)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 4)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_segmentation_model(config).to(device)
    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_config.get("amp", True)) and device.type == "cuda")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=int(config["model"].get("out_channels", 4)))])

    writer = SummaryWriter(log_dir=str(ROOT / "runs" / "segmentation"))
    best_metric = -1.0
    best_path = output_dir / "best_segresnet.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())

        epoch_loss /= max(1, len(train_loader))
        writer.add_scalar("loss/train", epoch_loss, epoch)

        should_validate = epoch % int(train_config.get("val_interval", 1)) == 0
        val_dice = -1.0
        if should_validate:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"val epoch {epoch}"):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        logits = sliding_window_inference(
                            images,
                            roi_size=roi_size,
                            sw_batch_size=int(train_config.get("sw_batch_size", 2)),
                            predictor=model,
                        )
                    preds = [post_pred(item) for item in decollate_batch(logits)]
                    dice_metric(y_pred=preds, y=decollate_batch(labels))
            val_dice = float(dice_metric.aggregate().item())
            dice_metric.reset()
            writer.add_scalar("dice/val_mean_no_background", val_dice, epoch)
            if val_dice > best_metric:
                best_metric = val_dice
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "dice": val_dice}, best_path)

        record = {"epoch": epoch, "train_loss": epoch_loss, "val_dice": val_dice, "best_dice": best_metric}
        history.append(record)
        print(record)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    writer.close()
    print(f"Best validation Dice: {best_metric:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
