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

    if backbone == "vit":
        # Patch size 8 is compatible with 200x200 input (200/8=25).
        img_size = image_size if isinstance(image_size, int) else image_size[0]
        model = timm.create_model(
            "vit_small_patch8_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=img_size,
        )
        return model, model.num_features

    raise ValueError("Unsupported backbone. Use 'resnet18' or 'vit'.")
