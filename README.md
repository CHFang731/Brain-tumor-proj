# Brain Tumor Project

Deep learning final project repository, organized into two parts:

- `tumor_detection/`: completed work for tumor detection / classification.
- `tumor_segmentation/`: active work for tumor segmentation (dataset, training, evaluation pipeline in place).

## Current Status

- Detection module is already available.
- Segmentation module now includes:
  - dataset download script
  - train/val/test split generation
  - U-Net training pipeline
  - held-out test evaluation pipeline
- Current best segmentation result:
  - Test Dice: `0.873212993`
  - Test IoU: `0.798736095`
  - Method: 8-model ref256 ensemble with Gaussian probability smoothing (`sigma=0.26`)

See `tumor_segmentation/README.md` for segmentation details and latest metrics.

## GitHub Inference Examples and Data

- Before/after inference images (segmentation + classification) and uploaded experiment JSON files are available in:
  - `github_assets/README.md`
- Segmentation experiment summary table:
  - `github_assets/experiment_data/segmentation/segmentation_experiment_summary.md`
  - `github_assets/experiment_data/segmentation/segmentation_experiment_summary.csv`
