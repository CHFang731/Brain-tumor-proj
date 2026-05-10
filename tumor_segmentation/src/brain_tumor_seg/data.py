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
from torchvision import transforms as T
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
        aggressive_aug: bool = False,
        conservative_aug: bool = False,
        train_repeat: int = 1,
        aug_qc_min_area_ratio: float = 0.45,
        aug_qc_max_area_ratio: float = 1.85,
        aug_qc_max_border_zero_increase: float = 0.28,
        elastic_aug: bool = False,
        elastic_prob: float = 0.2,
        elastic_alpha: float = 5.0,
        elastic_grid_size: int = 5,
    ) -> None:
        self.items = items
        self.image_size = int(image_size)
        self.training = training
        self.in_channels = int(in_channels)
        self.stronger_aug = stronger_aug
        self.aggressive_aug = aggressive_aug
        self.conservative_aug = conservative_aug
        self.train_repeat = max(1, int(train_repeat)) if training else 1
        self.aug_qc_min_area_ratio = float(aug_qc_min_area_ratio)
        self.aug_qc_max_area_ratio = float(aug_qc_max_area_ratio)
        self.aug_qc_max_border_zero_increase = float(aug_qc_max_border_zero_increase)
        self.elastic_aug = bool(elastic_aug)
        self.elastic_prob = float(elastic_prob)
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_grid_size = int(elastic_grid_size)

    def __len__(self) -> int:
        return len(self.items) * self.train_repeat

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

    def _random_resized_crop(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        scale: tuple[float, float],
        ratio: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top, left, height, width = T.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=height,
            width=width,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resized_crop(
            mask,
            top=top,
            left=left,
            height=height,
            width=width,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
            antialias=None,
        )
        return image, mask

    def _apply_cutout(self, image: torch.Tensor, max_frac: float = 0.22) -> torch.Tensor:
        _, height, width = image.shape
        cut_h = max(8, int(height * random.uniform(0.08, max_frac)))
        cut_w = max(8, int(width * random.uniform(0.08, max_frac)))
        y0 = random.randint(0, max(0, height - cut_h))
        x0 = random.randint(0, max(0, width - cut_w))
        image[:, y0 : y0 + cut_h, x0 : x0 + cut_w] = 0.0
        return image

    def _apply_elastic(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        grid_size = max(3, self.elastic_grid_size)
        disp = torch.randn(1, 2, grid_size, grid_size, dtype=image.dtype, device=image.device)
        disp = F.interpolate(disp, size=(height, width), mode="bicubic", align_corners=True).squeeze(0)
        disp = disp.permute(1, 2, 0)
        disp[..., 0] *= float(self.elastic_alpha) * 2.0 / max(1, width)
        disp[..., 1] *= float(self.elastic_alpha) * 2.0 / max(1, height)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=image.dtype, device=image.device),
            torch.linspace(-1.0, 1.0, width, dtype=image.dtype, device=image.device),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1) + disp
        image = F.grid_sample(
            image.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0),
            grid.unsqueeze(0),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        return image, mask

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_orig = image.clone()
        mask_orig = mask.clone()
        base_area = float(mask_orig.mean().item())

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

        if self.conservative_aug:
            if random.random() < 0.75:
                angle = random.uniform(-10.0, 10.0)
                translate_bound = int(self.image_size * 0.05)
                translate = [
                    int(random.uniform(-translate_bound, translate_bound)),
                    int(random.uniform(-translate_bound, translate_bound)),
                ]
                scale = random.uniform(0.95, 1.05)
                shear = [random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0)]
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0.0,
                )
            if random.random() < 0.18:
                image, mask = self._random_resized_crop(
                    image=image,
                    mask=mask,
                    scale=(0.88, 1.0),
                    ratio=(0.95, 1.05),
                )
            if random.random() < 0.18:
                noise = torch.randn_like(image) * random.uniform(0.006, 0.02)
                image = image + noise
            if random.random() < 0.2:
                gain = random.uniform(0.95, 1.05)
                bias = random.uniform(-0.04, 0.04)
                image = image * gain + bias
            if random.random() < 0.12:
                image = TF.adjust_gamma(image.clamp(0.0, 1.0), gamma=random.uniform(0.92, 1.08), gain=1.0)
            if random.random() < 0.1:
                image = TF.gaussian_blur(
                    image,
                    kernel_size=[3, 3],
                    sigma=[random.uniform(0.2, 0.6), random.uniform(0.2, 0.6)],
                )
            if self.elastic_aug and random.random() < self.elastic_prob:
                image, mask = self._apply_elastic(image, mask)

        if self.stronger_aug:
            if random.random() < 0.9:
                angle = random.uniform(-18.0, 18.0)
                translate_bound = int(self.image_size * 0.075)
                translate = [
                    int(random.uniform(-translate_bound, translate_bound)),
                    int(random.uniform(-translate_bound, translate_bound)),
                ]
                scale = random.uniform(0.88, 1.12)
                shear = [random.uniform(-6.0, 6.0), random.uniform(-6.0, 6.0)]
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0.0,
                )
            if random.random() < 0.35:
                image, mask = self._random_resized_crop(
                    image=image,
                    mask=mask,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1),
                )
            if random.random() < 0.3:
                noise = torch.randn_like(image) * random.uniform(0.01, 0.05)
                image = image + noise
            if random.random() < 0.3:
                gain = random.uniform(0.9, 1.1)
                bias = random.uniform(-0.1, 0.1)
                image = image * gain + bias
            if random.random() < 0.2:
                image = TF.adjust_gamma(image.clamp(0.0, 1.0), gamma=random.uniform(0.85, 1.25), gain=1.0)
            if random.random() < 0.15:
                image = TF.gaussian_blur(
                    image,
                    kernel_size=[random.choice([3, 5]), random.choice([3, 5])],
                    sigma=[random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)],
                )
            if random.random() < 0.15:
                image = TF.adjust_sharpness(image, sharpness_factor=random.uniform(0.7, 1.8))

        if self.aggressive_aug:
            if random.random() < 0.45:
                image, mask = self._random_resized_crop(
                    image=image,
                    mask=mask,
                    scale=(0.65, 1.0),
                    ratio=(0.85, 1.15),
                )
            if random.random() < 0.35:
                angle = random.uniform(-25.0, 25.0)
                translate_bound = int(self.image_size * 0.10)
                translate = [
                    int(random.uniform(-translate_bound, translate_bound)),
                    int(random.uniform(-translate_bound, translate_bound)),
                ]
                scale = random.uniform(0.82, 1.18)
                shear = [random.uniform(-9.0, 9.0), random.uniform(-9.0, 9.0)]
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0.0,
                )
            if random.random() < 0.25:
                image = self._apply_cutout(image)
            if random.random() < 0.3:
                noise = torch.randn_like(image) * random.uniform(0.03, 0.08)
                image = image + noise
            if random.random() < 0.25:
                image = TF.adjust_gamma(image.clamp(0.0, 1.0), gamma=random.uniform(0.75, 1.35), gain=1.0)
            if random.random() < 0.2:
                image = TF.gaussian_blur(
                    image,
                    kernel_size=[5, 5],
                    sigma=[random.uniform(0.4, 1.4), random.uniform(0.4, 1.4)],
                )
            if random.random() < 0.2:
                image = TF.adjust_sharpness(image, sharpness_factor=random.uniform(0.5, 2.2))

        image = image.clamp(0.0, 1.0)
        mask = (mask > 0.5).float()
        if not self._augmentation_quality_ok(
            image_before=image_orig,
            mask_before=mask_orig,
            image_after=image,
            mask_after=mask,
            base_area=base_area,
        ):
            return image_orig, mask_orig
        return image, mask

    @staticmethod
    def _border_zero_ratio(image: torch.Tensor, edge: int = 6) -> float:
        _, h, w = image.shape
        edge = max(1, min(edge, h // 2, w // 2))
        border = torch.zeros((h, w), dtype=torch.bool, device=image.device)
        border[:edge, :] = True
        border[-edge:, :] = True
        border[:, :edge] = True
        border[:, -edge:] = True
        vals = image[:, border]
        return float((vals <= 1e-4).float().mean().item())

    def _augmentation_quality_ok(
        self,
        image_before: torch.Tensor,
        mask_before: torch.Tensor,
        image_after: torch.Tensor,
        mask_after: torch.Tensor,
        base_area: float,
    ) -> bool:
        area_after = float(mask_after.mean().item())
        if area_after <= 1e-7:
            return False
        area_ratio = area_after / max(base_area, 1e-7)
        if area_ratio < self.aug_qc_min_area_ratio or area_ratio > self.aug_qc_max_area_ratio:
            return False

        border_before = self._border_zero_ratio(image_before)
        border_after = self._border_zero_ratio(image_after)
        if (border_after - border_before) > self.aug_qc_max_border_zero_increase:
            return False
        return True

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx % len(self.items)]
        image = self._load_image(item["image"])
        mask = self._load_mask(item["mask"])
        if self.training:
            image, mask = self._augment(image, mask)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + 1e-6)
        if self.in_channels > 1:
            image = image.repeat(self.in_channels, 1, 1)
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
