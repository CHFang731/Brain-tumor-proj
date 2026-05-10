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


def tversky_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - probs) * targets).sum(dim=1)
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - score.mean()


def _soft_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_boundary = _soft_boundary(probs)
    target_boundary = _soft_boundary(targets)
    pred_f = pred_boundary.flatten(1)
    target_f = target_boundary.flatten(1)
    inter = (pred_f * target_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + target_f.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float,
    pos_weight: torch.Tensor | None = None,
    focal_weight: float = 0.0,
    focal_gamma: float = 2.0,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    boundary_weight: float = 0.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, targets)
    base = bce_weight * bce + (1.0 - bce_weight) * dice
    if focal_weight > 0.0:
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = ((1.0 - p_t) ** focal_gamma * bce_raw).mean()
        base = (1.0 - focal_weight) * base + focal_weight * focal
    if tversky_weight > 0.0:
        tversky = tversky_loss_from_logits(
            logits,
            targets,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
        base = (1.0 - tversky_weight) * base + tversky_weight * tversky
    if boundary_weight > 0.0:
        boundary = boundary_dice_loss_from_logits(logits, targets)
        base = (1.0 - boundary_weight) * base + boundary_weight * boundary
    return base


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
