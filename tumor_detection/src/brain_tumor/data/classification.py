from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ClassificationItem:
    image: str
    label: int
    class_name: str


def discover_classification_items(
    dataset_root: str | Path,
    split: str,
    class_names: list[str] | None = None,
) -> list[ClassificationItem]:
    root = Path(dataset_root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Classification split not found: {split_dir}")

    classes = class_names or sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    items: list[ClassificationItem] = []
    for label, class_name in enumerate(classes):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory missing: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(ClassificationItem(str(image_path), label, class_name))
    if not items:
        raise RuntimeError(f"No classification images found under {split_dir}")
    return items


def as_monai_dicts(items: list[ClassificationItem]) -> list[dict[str, str | int]]:
    return [{"image": item.image, "label": item.label} for item in items]
