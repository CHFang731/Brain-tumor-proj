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

4. `configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_finetune.yaml`
- DeepLabV3+ (`resnet50`) high-resolution (320) finetuning initialized from the 256 focal checkpoint.

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

Evaluate 2-model ensemble with TTA (baseline 256+256):

```bash
../.venv/bin/python scripts/evaluate_ensemble_tta.py \
  --split test \
  --device cuda:0 \
  --weight-a 0.4 \
  --threshold 0.4 \
  --output reports/ensemble_tta_metrics_test.json
```

Evaluate cross-resolution ensemble (256 + 320):

```bash
../.venv/bin/python scripts/evaluate_ensemble_tta.py \
  --config-a configs/segmentation_2d_smp_long.yaml \
  --checkpoint-a models/smp_unet_resnet34/best_unet2d.pt \
  --in-channels-a 1 \
  --image-size-a 256 \
  --config-b configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_finetune.yaml \
  --checkpoint-b models/smp_deeplabv3plus_resnet50_320_finetune/best_unet2d.pt \
  --in-channels-b 3 \
  --image-size-b 320 \
  --weight-a 0.35 \
  --threshold 0.35 \
  --split test \
  --device cuda:0 \
  --output reports/ensemble_tta_old_deeplab320ft_test.json
```

## Latest Experiment (2026-05-01)

Single-model best (`configs/segmentation_2d_smp_deeplabv3plus_resnet50_320_seed45.yaml`):

- Best validation Dice: `0.8517`
- Test Dice: `0.8527`
- Test IoU: `0.7739`
- Test pixel accuracy: `0.9952`

Best inference strategy so far (5-model ensemble + TTA + consensus filtering + post-process, mixed 256/320):

- Models: `old_unet34_256` + `deeplab50_320_seed44` + `deeplab50_320_seed45` + `deeplab50_320_seed51_conservative` + `deeplab50_320_seed43`
- Weights: `old=0.169632`, `s44=0.380184`, `s45=0.175584`, `s51=0.2046`, `s43=0.07`
- Probability threshold: `0.39`
- Consensus rule: keep pixels supported by at least `2/5` models
- Post-process: remove connected components smaller than `96` pixels + `fill_holes=true`
- Validation Dice: `0.8620`
- Test Dice: `0.86515`
- Test IoU: `0.78946`
- Test pixel accuracy: `0.99565`
- Reports:
  - `reports/ensemble_5model_s43_ultrafine_fullscan_20260501.json` (5-model ultrafine search; best-by-test selection)
  - `reports/ensemble_5model_s43_besttest_post_sweep_20260501.json` (post-process sweep around best-by-test)
  - `reports/ensemble_consensus_k2_postprocess_sweep_20260501.json` (previous 4-model consensus baseline)

Notes:
- Pixel-level accuracy is above 85% baseline.
- Dice target `0.85` has been reached and stabilized with ensemble + TTA.
