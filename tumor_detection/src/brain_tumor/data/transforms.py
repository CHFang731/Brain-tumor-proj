from __future__ import annotations


def segmentation_transforms(roi_size: tuple[int, int, int], training: bool):
    from monai.transforms import (
        AsDiscreted,
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        MapLabelValued,
        NormalizeIntensityd,
        Orientationd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
    )

    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if training:
        base.extend(
            [
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="label",
                    spatial_size=roi_size,
                    pos=1,
                    neg=1,
                    num_samples=2,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
            ]
        )
    base.extend(
        [
            EnsureTyped(keys=keys),
            AsDiscreted(keys="label", to_onehot=4),
        ]
    )
    return Compose(base)


def segmentation_inference_transforms():
    from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, Orientationd

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )


def classification_transforms(image_size: int, training: bool):
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        RandFlipd,
        RandRotate90d,
        Resized,
        ScaleIntensityd,
    )

    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size)),
    ]
    if training:
        transforms.extend(
            [
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=["image"], prob=0.25, max_k=3),
            ]
        )
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)
