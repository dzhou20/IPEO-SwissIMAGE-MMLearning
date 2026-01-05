from __future__ import annotations

import torch
import torch.nn as nn

from config import NUM_CLASSES
from src.models.backbone import build_backbone


class ImageOnlyModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        image_size: int | tuple[int, int],
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone, pretrained, image_size)
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
        image_size: int | tuple[int, int],
        num_classes: int = NUM_CLASSES,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone, pretrained, image_size)

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


class TabularOnlyModel(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        num_classes: int = NUM_CLASSES,
        hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        return self.mlp(tabular)


class TwoStageImageModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        image_size: int | tuple[int, int],
        num_level1_classes: int,
        num_level2_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone, pretrained, image_size)
        self.head_level1 = nn.Linear(feature_dim, num_level1_classes)
        self.head_level2 = nn.Linear(feature_dim, num_level2_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(images)
        return self.head_level1(feats), self.head_level2(feats)


class TwoStageEarlyFusionModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        tabular_dim: int,
        image_size: int | tuple[int, int],
        num_level1_classes: int,
        num_level2_classes: int = NUM_CLASSES,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_backbone(backbone, pretrained, image_size)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fused_dim = feature_dim + fusion_hidden
        self.head_level1 = nn.Linear(fused_dim, num_level1_classes)
        self.head_level2 = nn.Linear(fused_dim, num_level2_classes)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_feats = self.encoder(images)
        tab_feats = self.tabular_mlp(tabular)
        fused = torch.cat([img_feats, tab_feats], dim=1)
        return self.head_level1(fused), self.head_level2(fused)


class TwoStageTabularModel(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        num_level1_classes: int,
        num_level2_classes: int = NUM_CLASSES,
        hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_level1 = nn.Linear(256, num_level1_classes)
        self.head_level2 = nn.Linear(256, num_level2_classes)

    def forward(self, tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.mlp(tabular)
        return self.head_level1(feats), self.head_level2(feats)
