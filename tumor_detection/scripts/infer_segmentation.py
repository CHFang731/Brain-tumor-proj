#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import configure_runtime, ensure_dir, load_config
from brain_tumor.data.brats import MODALITIES
from brain_tumor.data.transforms import segmentation_inference_transforms
from brain_tumor.models.segmentation import create_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation inference on one BraTS-style subject.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--checkpoint", default="models/segmentation/best_segresnet.pt")
    parser.add_argument("--subject-dir", required=True, help="Folder containing *_t1, *_t1ce, *_t2, *_flair NIfTI files.")
    parser.add_argument("--output-dir", default="reports/inference")
    return parser.parse_args()


def modality_paths(subject_dir: Path) -> list[str]:
    paths: list[str] = []
    for modality in MODALITIES:
        matches = sorted(subject_dir.glob(f"*_{modality}.nii*"))
        if not matches:
            raise FileNotFoundError(f"Missing {modality} file in {subject_dir}")
        paths.append(str(matches[0]))
    return paths


def main() -> None:
    configure_runtime(ROOT)
    import nibabel as nib
    import torch
    from monai.inferers import sliding_window_inference

    args = parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" in checkpoint:
        config = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_segmentation_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    subject_dir = Path(args.subject_dir)
    image_paths = modality_paths(subject_dir)
    transform = segmentation_inference_transforms()
    item = transform({"image": image_paths})
    image = item["image"].unsqueeze(0).to(device)
    roi_size = tuple(int(v) for v in config["training"]["roi_size"])

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=roi_size,
            sw_batch_size=int(config["training"].get("sw_batch_size", 2)),
            predictor=model,
        )
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype("uint8")
    pred[pred == 3] = 4

    reference = nib.load(image_paths[0])
    output_dir = ensure_dir(args.output_dir)
    output_path = output_dir / f"{subject_dir.name}_prediction_mask.nii.gz"
    nib.save(nib.Nifti1Image(pred, reference.affine, reference.header), output_path)
    print(f"Saved prediction mask: {output_path}")


if __name__ == "__main__":
    main()
