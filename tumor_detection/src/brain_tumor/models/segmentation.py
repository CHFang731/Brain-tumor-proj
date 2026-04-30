from __future__ import annotations


def create_segmentation_model(config: dict):
    from brain_tumor.config import configure_runtime

    configure_runtime()
    from monai.networks.nets import SegResNet, SwinUNETR, UNet

    model_config = config.get("model", {})
    name = model_config.get("name", "segresnet").lower()
    in_channels = int(model_config.get("in_channels", 4))
    out_channels = int(model_config.get("out_channels", 4))

    if name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=int(model_config.get("init_filters", 32)),
            dropout_prob=float(model_config.get("dropout_prob", 0.0)),
        )
    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(model_config.get("channels", [16, 32, 64, 128, 256])),
            strides=tuple(model_config.get("strides", [2, 2, 2, 2])),
            num_res_units=int(model_config.get("num_res_units", 2)),
        )
    if name == "swinunetr":
        roi_size = tuple(config.get("training", {}).get("roi_size", [128, 128, 128]))
        return SwinUNETR(
            img_size=roi_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=int(model_config.get("feature_size", 48)),
            use_checkpoint=bool(model_config.get("use_checkpoint", True)),
        )
    raise ValueError(f"Unsupported segmentation model: {name}")
