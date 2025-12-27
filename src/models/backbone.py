from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torchvision.models as models
import timm


def build_backbone(backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if backbone == "vit":
        # Patch size 8 is compatible with 200x200 input (200/8=25).
        model = timm.create_model(
            "vit_small_patch8_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, model.num_features

    raise ValueError("Unsupported backbone. Use 'resnet18' or 'vit'.")
