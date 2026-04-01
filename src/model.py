"""
model.py
────────
Supports multiple backbone architectures for ensemble training.

Supported models:
  - efficientnet_b4  (best accuracy, ~19M params)
  - resnet50         (fast, different feature extraction)

Usage:
  from src.model import build_model
  model = build_model(arch='efficientnet_b4')
  model = build_model(arch='resnet50')
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B4_Weights,
    ResNet50_Weights,
)


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier.
    Supports EfficientNetB4 and ResNet50 backbones.
    """

    def __init__(
        self,
        arch       : str   = 'efficientnet_b4',
        num_classes: int   = 5,
        dropout    : float = 0.4,
        freeze     : bool  = True,
    ):
        super().__init__()
        self.arch = arch

        # ── Backbone ──────────────────────────────────────────────────────────
        if arch == 'efficientnet_b4':
            self.backbone   = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_features     = self.backbone.classifier[1].in_features   # 1792
            self.backbone.classifier = nn.Identity()

        elif arch == 'resnet50':
            self.backbone   = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            in_features     = self.backbone.fc.in_features              # 2048
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(f'Unsupported arch: {arch}. Choose efficientnet_b4 or resnet50')

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f'[{self.arch}] Backbone frozen ✅ — Phase 1')

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f'[{self.arch}] Backbone unfrozen ✅ — Phase 2')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ─────────────────────────────────────────────────────────────────────────────
# Factory + helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    arch       : str   = 'efficientnet_b4',
    num_classes: int   = 5,
    dropout    : float = 0.4,
    freeze     : bool  = True,
) -> DRClassifier:
    return DRClassifier(arch=arch, num_classes=num_classes,
                        dropout=dropout, freeze=freeze)


def count_parameters(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {'trainable': trainable, 'frozen': frozen, 'total': trainable + frozen}


def get_optimizer(model: nn.Module, phase: int = 1, arch: str = 'efficientnet_b4'):
    """
    Phase 1: high LR, head only
    Phase 2: differential LR — backbone gets 10x lower than head
             ResNet gets slightly higher LR than EfficientNet (it needs more nudging)
    """
    if phase == 1:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )

    # Phase 2 — conservative LRs to prevent destroying pretrained weights
    backbone_lr = 1e-6 if arch == 'efficientnet_b4' else 5e-6
    head_lr     = 1e-5 if arch == 'efficientnet_b4' else 5e-5

    return torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.head.parameters(),     'lr': head_lr},
    ], weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy  = torch.randn(4, 3, 224, 224).to(device)

    for arch in ['efficientnet_b4', 'resnet50']:
        print(f'\n── {arch} ──')
        model  = build_model(arch=arch, freeze=True).to(device)
        params = count_parameters(model)
        out    = model(dummy)
        print(f'  Trainable : {params["trainable"]:,}')
        print(f'  Total     : {params["total"]:,}')
        print(f'  Output    : {out.shape}')

    print('\nmodel.py ✅')
