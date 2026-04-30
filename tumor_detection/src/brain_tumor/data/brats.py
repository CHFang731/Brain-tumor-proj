from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODALITIES = ("t1", "t1ce", "t2", "flair")


@dataclass(frozen=True)
class BratsSubject:
    subject_id: str
    image: list[str]
    label: str | None

    def as_monai_dict(self) -> dict[str, str | list[str]]:
        item: dict[str, str | list[str]] = {"image": self.image, "subject_id": self.subject_id}
        if self.label is not None:
            item["label"] = self.label
        return item


def _find_one(subject_dir: Path, suffixes: Iterable[str]) -> Path | None:
    for suffix in suffixes:
        matches = sorted(subject_dir.glob(f"*_{suffix}.nii*"))
        if matches:
            return matches[0]
    return None


def discover_brats_subjects(root: str | Path, require_label: bool = True) -> list[BratsSubject]:
    """Discover BraTS-style subject folders.

    Expected files per subject include t1, t1ce, t2, flair, and optionally seg:
    ``BraTS-GLI-00000-000_t1.nii.gz``.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"BraTS root not found: {root_path}. Place BraTS subject folders under this path."
        )

    subjects: list[BratsSubject] = []
    candidate_dirs = [p for p in root_path.iterdir() if p.is_dir()]
    for subject_dir in sorted(candidate_dirs):
        image_paths: list[str] = []
        missing: list[str] = []
        for modality in MODALITIES:
            found = _find_one(subject_dir, (modality,))
            if found is None:
                missing.append(modality)
            else:
                image_paths.append(str(found))

        label_path = _find_one(subject_dir, ("seg", "mask", "label"))
        if missing or (require_label and label_path is None):
            continue

        subjects.append(
            BratsSubject(
                subject_id=subject_dir.name,
                image=image_paths,
                label=str(label_path) if label_path is not None else None,
            )
        )

    if not subjects:
        label_msg = " with segmentation masks" if require_label else ""
        raise RuntimeError(f"No complete BraTS subjects{label_msg} found under {root_path}")
    return subjects


def make_or_load_split(
    root: str | Path,
    split_json: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, str | list[str]]]]:
    split_path = Path(split_json)
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    subjects = discover_brats_subjects(root, require_label=True)
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    split = {
        "train": [subject.as_monai_dict() for subject in shuffled[val_count:]],
        "val": [subject.as_monai_dict() for subject in shuffled[:val_count]],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    return split
