# Tumor Segmentation Module

This module implements brain tumor semantic segmentation with a strict train/val/test split.

## Dataset

- Source: https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation
- Content: 3064 MRI image-mask pairs (`images/*.png`, `masks/*.png`), binary masks.
- Local cache path used in this project:
  - `../kagglehub_cache/datasets/nikhilroxtomar/brain-tumor-segmentation/versions/1`

## Data Split Policy (No Test Leakage)

- Split file: `data/processed/split_2d.json`
- Current split counts:
  - train: 2144
  - val: 459
  - test: 461
- `scripts/train_segmentation.py` only uses `train` for optimization and `val` for model selection.
- `test` is only used by `scripts/evaluate_segmentation.py` after training.

## Models

Two model configurations are included:

1. `configs/segmentation_2d.yaml`
- Custom 2D U-Net.

2. `configs/segmentation_2d_smp.yaml`
- U-Net with pretrained encoder (`segmentation_models_pytorch`, `resnet34`).

## Commands

Download dataset:

```bash
cd /home/fangdog/brain_tumor/tumor_segmentation
../.venv/bin/python scripts/download_dataset.py --dataset nikhilroxtomar/brain-tumor-segmentation
```

Prepare split:

```bash
../.venv/bin/python scripts/prepare_segmentation_split.py --config configs/segmentation_2d_smp.yaml --force
```

Train:

```bash
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python scripts/train_segmentation.py \
  --config configs/segmentation_2d_smp.yaml \
  --device cuda:0
```

Evaluate on held-out test split:

```bash
../.venv/bin/python scripts/evaluate_segmentation.py \
  --config configs/segmentation_2d_smp.yaml \
  --checkpoint models/smp_unet_resnet34/best_unet2d.pt \
  --split test \
  --device cuda:0
```

## Latest Experiment (2026-04-30)

Using `configs/segmentation_2d_smp.yaml`:

- Best validation Dice: `0.7924`
- Test Dice: `0.7886`
- Test IoU: `0.6955`
- Test pixel accuracy: `0.9926`

Notes:
- Pixel-level accuracy is above 85% baseline.
- Dice is currently below 0.85; further tuning (higher input resolution, stronger augmentations, and longer training) is still needed to push Dice beyond 0.85.
