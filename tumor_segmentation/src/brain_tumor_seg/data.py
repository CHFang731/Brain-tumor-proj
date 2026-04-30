from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class SegPair:
    image: str
    mask: str
    id: str

    def as_dict(self) -> dict[str, str]:
        return {"image": self.image, "mask": self.mask, "id": self.id}


class BrainTumorSegDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, str]],
        image_size: int,
        training: bool,
        in_channels: int = 1,
        stronger_aug: bool = False,
    ) -> None:
        self.items = items
        self.image_size = int(image_size)
        self.training = training
        self.in_channels = int(in_channels)
        self.stronger_aug = stronger_aug

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).unsqueeze(0)
        ten = F.interpolate(
            ten.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mean = ten.mean()
        std = ten.std()
        ten = (ten - mean) / (std + 1e-6)
        if self.in_channels > 1:
            ten = ten.repeat(self.in_channels, 1, 1)
        return ten

    def _load_mask(self, path: str) -> torch.Tensor:
        m = Image.open(path).convert("L")
        arr = np.asarray(m, dtype=np.float32)
        arr = (arr > 127).astype(np.float32)
        ten = torch.from_numpy(arr).unsqueeze(0)
        ten = F.interpolate(
            ten.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)
        return ten

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() < 0.2:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        if self.stronger_aug:
            angle = random.uniform(-15.0, 15.0)
            translate = [int(random.uniform(-20, 20)), int(random.uniform(-20, 20))]
            scale = random.uniform(0.9, 1.1)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
            if random.random() < 0.3:
                noise = torch.randn_like(image) * random.uniform(0.01, 0.05)
                image = image + noise
            if random.random() < 0.3:
                gain = random.uniform(0.9, 1.1)
                bias = random.uniform(-0.1, 0.1)
                image = image * gain + bias
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx]
        image = self._load_image(item["image"])
        mask = self._load_mask(item["mask"])
        if self.training:
            image, mask = self._augment(image, mask)
        return {"image": image, "mask": mask, "id": item["id"]}


def discover_pairs(dataset_root: str | Path) -> list[SegPair]:
    root = Path(dataset_root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders not found under {root}. Required: images/ and masks/."
        )

    images = {p.name: p for p in images_dir.glob("*.png")}
    masks = {p.name: p for p in masks_dir.glob("*.png")}
    common = sorted(images.keys() & masks.keys(), key=lambda name: int(Path(name).stem))
    if not common:
        raise RuntimeError(f"No matched image/mask pairs found in {root}")

    return [
        SegPair(
            image=str(images[name].resolve()),
            mask=str(masks[name].resolve()),
            id=Path(name).stem,
        )
        for name in common
    ]


def make_or_load_split(
    dataset_root: str | Path,
    split_json: str | Path,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    split_path = Path(split_json)
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        return split

    if train_fraction <= 0 or val_fraction <= 0 or (train_fraction + val_fraction) >= 1:
        raise ValueError("train_fraction and val_fraction must be > 0 and sum to < 1")

    pairs = discover_pairs(dataset_root)
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_items = shuffled[:n_train]
    val_items = shuffled[n_train : n_train + n_val]
    test_items = shuffled[n_train + n_val :]

    split = {
        "train": [x.as_dict() for x in train_items],
        "val": [x.as_dict() for x in val_items],
        "test": [x.as_dict() for x in test_items],
        "meta": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "counts": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
        },
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    return split
