from __future__ import annotations

import torch
import torch.nn as nn

from config import NUM_CLASSES
from src.models.backbone import build_backbone


class ImageOnlyModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone, pretrained)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        return self.head(feats)


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        tabular_dim: int,
        num_classes: int = NUM_CLASSES,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone, pretrained)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim + fusion_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feats = self.encoder(images)
        tab_feats = self.tabular_mlp(tabular)
        fused = torch.cat([img_feats, tab_feats], dim=1)
        return self.head(fused)
