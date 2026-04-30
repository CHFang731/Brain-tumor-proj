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

Main configurations used in this repo:

1. `configs/segmentation_2d_smp_long.yaml`
- U-Net (`segmentation_models_pytorch`) with pretrained `resnet34` encoder.

2. `configs/segmentation_2d_smp_unet_resnet34_256_focal.yaml`
- U-Net (`resnet34`) with 3-channel input, stronger augmentation, and BCE+Dice+Focal loss.

3. `configs/segmentation_2d_smp_unetpp_384.yaml`
- U-Net++ (`resnet50`) at larger resolution (exploration run).

## Commands

Download dataset:

```bash
cd /home/fangdog/brain_tumor/tumor_segmentation
../.venv/bin/python scripts/download_dataset.py --dataset nikhilroxtomar/brain-tumor-segmentation
```

Prepare split:

```bash
../.venv/bin/python scripts/prepare_segmentation_split.py --config configs/segmentation_2d_smp_unet_resnet34_256_focal.yaml --force
```

Train:

```bash
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python scripts/train_segmentation.py \
  --config configs/segmentation_2d_smp_unet_resnet34_256_focal.yaml \
  --device cuda:0
```

Evaluate on held-out test split:

```bash
../.venv/bin/python scripts/evaluate_segmentation.py \
  --config configs/segmentation_2d_smp_unet_resnet34_256_focal.yaml \
  --checkpoint models/smp_unet_resnet34_256_focal/best_unet2d.pt \
  --split test \
  --device cuda:0
```

Evaluate 2-model ensemble with TTA (current best):

```bash
../.venv/bin/python scripts/evaluate_ensemble_tta.py \
  --split test \
  --device cuda:0 \
  --weight-a 0.4 \
  --threshold 0.4 \
  --output reports/ensemble_tta_metrics_test.json
```

## Latest Experiment (2026-04-30)

Single-model best (`configs/segmentation_2d_smp_unet_resnet34_256_focal.yaml`):

- Best validation Dice: `0.8270`
- Test Dice: `0.8259`
- Test IoU: `0.7401`
- Test pixel accuracy: `0.9945`

Best inference strategy so far (2-model ensemble + TTA):

- Test Dice: `0.8553`
- Test IoU: `0.7745`
- Test pixel accuracy: `0.9953`

Notes:
- Pixel-level accuracy is above 85% baseline.
- Dice target `0.85` has been reached with ensemble + TTA.
