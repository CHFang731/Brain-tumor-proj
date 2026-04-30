# Tumor Detection Module

This module contains the current implementation for brain tumor detection tasks.

## Scope

- Current focus: tumor detection / classification pipeline.
- Segmentation work is intentionally separated and will be developed under `../tumor_segmentation/` later.

## Setup

```bash
cd /home/fangdog/brain_tumor/tumor_detection
source ../.venv/bin/activate
python -m pip install -r requirements.txt
```

## Detection Training

```bash
python scripts/train_classification.py --config configs/classification.yaml
```

Baseline v1 training on a selected GPU:

```bash
python scripts/train_classification.py --config configs/classification_v1.yaml --device cuda:1
```

## Detection Inference

```bash
python scripts/infer_classification.py \
  --config configs/classification_v1.yaml \
  --checkpoint models/classification/v1/best_classifier.pt \
  --image data/raw/kaggle_classification/brain-tumor-mri-dataset/Testing/meningioma/Te-me_272.jpg \
  --output reports/classification_prediction_v1.json \
  --device cuda:1
```

## Notes

- `data/raw/kaggle_classification/` should contain the image classification dataset.
- Large artifacts under `models/`, `runs/`, and `reports/` are ignored by Git.
