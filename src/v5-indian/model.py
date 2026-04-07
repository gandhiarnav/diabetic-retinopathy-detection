"""
model.py
────────
Two model architectures from the notebook:

  ConvNextModel    ← WINNER — kappa 0.9537 in 5 epochs
  MultimodalModel  ← EfficientNetB0 + CLIP text fusion

Both output dual heads:
  dr_head    → DR grade (0–4)
  edema_head → Macular edema risk (0–2)
"""

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_DR_CLASSES, NUM_EDEMA_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# ConvNextModel — the high-accuracy model (kappa 0.95)
# ─────────────────────────────────────────────────────────────────────────────

class ConvNextModel(nn.Module):
    """
    ConvNeXt-Tiny backbone with dual output heads.

    Why ConvNeXt beats EfficientNet here:
      ConvNeXt uses modern design principles (large kernels, LayerNorm,
      GELU activations) that make it better at capturing the subtle
      texture differences between DR grades than older architectures.

    Architecture:
      ConvNeXt-Tiny backbone (pretrained ImageNet)
            ↓
      Flatten (B, 768)
            ↓
      ┌─────────────┬──────────────┐
      DR head       Edema head
      Linear(768,5) Linear(768,3)
    """

    def __init__(
        self,
        num_dr_classes   : int = NUM_DR_CLASSES,
        num_edema_classes: int = NUM_EDEMA_CLASSES,
    ):
        super().__init__()

        # ConvNeXt-Tiny pretrained on ImageNet
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        # Replace classifier with flatten only — we add our own heads
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),   # (B, 768, 1, 1) → (B, 768)
        )

        self.dr_head    = nn.Linear(768, num_dr_classes)
        self.edema_head = nn.Linear(768, num_edema_classes)

    def forward(self, image: torch.Tensor, text_features=None):
        """text_features is accepted but ignored — image-only model."""
        out = self.backbone(image)
        return self.dr_head(out), self.edema_head(out)


# ─────────────────────────────────────────────────────────────────────────────
# MultimodalModel — EfficientNetB0 + CLIP text fusion
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalModel(nn.Module):
    """
    EfficientNetB0 image backbone fused with CLIP text embeddings from captions.

    Architecture:
      Image branch : EfficientNetB0 → (B, 1280)
      Text branch  : Linear(512→128) + ReLU + BN → (B, 128)
      Fusion       : Concat → Linear(1408→256) + ReLU + Dropout(0.3)
            ↓
      ┌─────────────┬──────────────┐
      DR head       Edema head
      Linear(256,5) Linear(256,3)

    When use_captions=False in the dataset, text_features are zeros
    and the model works as image-only.
    """

    def __init__(
        self,
        num_dr_classes   : int = NUM_DR_CLASSES,
        num_edema_classes: int = NUM_EDEMA_CLASSES,
    ):
        super().__init__()

        # EfficientNetB0 backbone
        self.image_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.image_model.classifier = nn.Identity()  # output: (B, 1280)

        # Text branch — processes CLIP 512-dim embeddings
        self.text_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        # Fusion — combines image + text
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.dr_head    = nn.Linear(256, num_dr_classes)
        self.edema_head = nn.Linear(256, num_edema_classes)

    def forward(self, image: torch.Tensor, text_features: torch.Tensor):
        img_out = self.image_model(image)           # (B, 1280)
        txt_out = self.text_branch(text_features)   # (B, 128)
        fused   = self.fusion(
            torch.cat([img_out, txt_out], dim=1)    # (B, 1408)
        )
        return self.dr_head(fused), self.edema_head(fused)
