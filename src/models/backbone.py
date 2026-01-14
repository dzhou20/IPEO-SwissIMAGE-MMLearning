from __future__ import annotations

from typing import Tuple, Union

import torch.nn as nn
import torchvision.models as models
import timm


def build_backbone(
    backbone: str,
    pretrained: bool,
    image_size: Union[int, Tuple[int, int]],
) -> Tuple[nn.Module, int]:
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    # ---------------- ConvNeXt Tiny ----------------
    if backbone == "convnext_tiny":
        model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,        # output features
            global_pool="avg",
        )
        return model, model.num_features

    # ---------------- EfficientNet-B0 ----------------
    if backbone == "efficientnet_b0":
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, model.num_features

    raise ValueError("Unsupported backbone. Use 'resnet18', convnext_tiny' or 'efficientnet_b0'.")
