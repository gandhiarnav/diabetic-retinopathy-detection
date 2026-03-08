"""
model.py
────────
EfficientNetB4-based classifier for Diabetic Retinopathy grading (0-4).

Architecture:
  - Backbone : EfficientNetB4 pretrained on ImageNet
  - Head     : GlobalAvgPool → Dropout → FC(256) → Dropout → FC(5)

Two-phase training strategy:
  - Phase 1: Backbone frozen  → train head only      (fast, ~10 epochs)
  - Phase 2: Backbone unfrozen → fine-tune everything (slow, ~20 epochs)

Usage:
  from src.model import build_model, count_parameters

  model = build_model(num_classes=5, dropout=0.4)
  model = model.to(device)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier built on EfficientNetB4.

    Args:
        num_classes : Number of output classes (5 for DR grading 0-4)
        dropout     : Dropout probability in the classification head
        freeze      : If True, freeze the backbone (Phase 1 training)
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.4,
        freeze: bool = True,
    ):
        super().__init__()

        # ── Backbone: EfficientNetB4 pretrained on ImageNet ───────────────────
        self.backbone = models.efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # Get the number of features output by the backbone
        in_features = self.backbone.classifier[1].in_features   # 1792 for B4

        # Remove the original classifier head
        self.backbone.classifier = nn.Identity()

        # ── Custom classification head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Freeze backbone if Phase 1
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1 — train head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print('Backbone frozen ✅  — training head only (Phase 1)')

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (Phase 2 — fine-tune everything)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print('Backbone unfrozen ✅ — fine-tuning all layers (Phase 2)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # (B, 1792)
        out      = self.head(features)   # (B, num_classes)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int = 5,
    dropout: float = 0.4,
    freeze: bool = True,
) -> DRClassifier:
    """
    Convenience factory function.

    Args:
        num_classes : 5 for DR grading (0-4)
        dropout     : Dropout rate in head (0.3-0.5 recommended)
        freeze      : True for Phase 1, False for Phase 2

    Returns:
        DRClassifier instance
    """
    model = DRClassifier(num_classes=num_classes, dropout=dropout, freeze=freeze)
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable vs frozen parameters.

    Returns:
        dict with 'trainable', 'frozen', 'total' counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen
    return {
        'trainable': trainable,
        'frozen'   : frozen,
        'total'    : total,
    }


def get_optimizer(model: nn.Module, phase: int = 1) -> torch.optim.Optimizer:
    """
    Returns the recommended optimizer for each training phase.

    Phase 1: Higher LR — only head params are trained
    Phase 2: Low LR   — fine-tuning, use differential learning rates
               backbone gets 10x lower LR than head

    Args:
        model : DRClassifier instance
        phase : 1 or 2

    Returns:
        torch.optim.AdamW optimizer
    """
    if phase == 1:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )
    else:
        # Differential learning rates: backbone gets 10x lower LR
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.head.parameters(),     'lr': 1e-4},
        ], weight_decay=1e-4)

    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run directly to verify model builds correctly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── Phase 1 model ─────────────────────────────────────────────────────────
    print('── Phase 1 (frozen backbone) ──')
    model = build_model(num_classes=5, dropout=0.4, freeze=True)
    model = model.to(device)

    params = count_parameters(model)
    print(f'  Trainable params : {params["trainable"]:,}')
    print(f'  Frozen params    : {params["frozen"]:,}')
    print(f'  Total params     : {params["total"]:,}')

    # Forward pass with a dummy batch
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f'  Output shape     : {out.shape}')     # (4, 5)
    print()

    # ── Phase 2 model ─────────────────────────────────────────────────────────
    print('── Phase 2 (unfrozen backbone) ──')
    model.unfreeze_backbone()
    params = count_parameters(model)
    print(f'  Trainable params : {params["trainable"]:,}')
    print(f'  Frozen params    : {params["frozen"]:,}')

    optimizer = get_optimizer(model, phase=2)
    print(f'  Optimizer LR groups:')
    for i, g in enumerate(optimizer.param_groups):
        print(f'    Group {i}: lr = {g["lr"]}')

    print('\nmodel.py ✅')