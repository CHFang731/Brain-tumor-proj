from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

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
