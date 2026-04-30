from __future__ import annotations


def create_classification_model(config: dict):
    from brain_tumor.config import configure_runtime

    configure_runtime()
    from monai.networks.nets import DenseNet121, EfficientNetBN

    model_config = config.get("model", {})
    name = model_config.get("name", "densenet121").lower()
    spatial_dims = int(model_config.get("spatial_dims", 2))
    in_channels = int(model_config.get("in_channels", 3))
    num_classes = int(model_config.get("num_classes", 4))

    if name == "densenet121":
        return DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
        )
    if name.startswith("efficientnet"):
        return EfficientNetBN(
            model_name=name,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported classification model: {name}")
