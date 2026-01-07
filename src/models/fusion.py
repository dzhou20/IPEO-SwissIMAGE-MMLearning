from __future__ import annotations

import torch
import torch.nn as nn

from config import NUM_CLASSES
from src.models.backbone import build_backbone

class TabularOnlyModel(nn.Module):
    def __init__(
        self, 
        tabular_dim: int,           
        num_classes: int = NUM_CLASSES, 
        hidden_dim: int = 128,    
        dropout: float = 0.2     
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, tabular: torch.Tensor, images: torch.Tensor = None) -> torch.Tensor:
        feats = self.mlp(tabular)
        return self.head(feats)

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

class GatedFusionModel(nn.Module):
    """
    Extends standard early fusion with a context-gating mechanism where tabular embeddings 
    dynamically re-weight image features, using environmental context to emphasize relevant 
    visual patterns before concatenation.
    """
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        tabular_dim: int,
        image_size: int | tuple[int, int],
        num_classes: int = NUM_CLASSES,
        tabular_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        # Image Backbone
        self.encoder, img_feature_dim = build_backbone(backbone, pretrained, image_size)

        # Tabular Encoder (Context Extractor)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention Gate Network
        # Projects tabular features to the SAME dimension as image features.
        self.attention_gate = nn.Sequential(
            nn.Linear(tabular_hidden, img_feature_dim),
            nn.Sigmoid()
        )

        # Final Classifier Head
        fusion_output_dim = img_feature_dim + tabular_hidden
        
        self.head = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feats = self.encoder(images)
        tab_feats = self.tabular_mlp(tabular)
        attention_weights = self.attention_gate(tab_feats)
        img_feats_weighted = img_feats * attention_weights
        combined = torch.cat([img_feats_weighted, tab_feats], dim=1)
        return self.head(combined)
    
class LateFusionModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        tabular_dim: int,
        image_size: int | tuple[int, int],
        num_classes: int = NUM_CLASSES,
        tabular_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.encoder, img_dim = build_backbone(backbone, pretrained, image_size)
        self.img_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(img_dim, num_classes)
        )

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tabular_head = nn.Sequential(
            nn.Linear(tabular_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.fusion_weight = nn.Parameter(torch.zeros(1)) 

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feats = self.encoder(images)
        logits_img = self.img_head(img_feats)
        
        tab_feats = self.tabular_mlp(tabular)
        logits_tab = self.tabular_head(tab_feats)
        
        alpha = torch.sigmoid(self.fusion_weight)
        logits_final = alpha * logits_img + (1 - alpha) * logits_tab
        
        return logits_final