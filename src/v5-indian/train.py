"""
train.py
────────
Trains both models from the notebook and compares them.

Results reproduced from notebook:
  image_only (EfficientNetB0) → Kappa 0.8578
  convnext                    → Kappa 0.9537   ← use this one

Usage on Kaggle:
  from train import run_all
  run_all()

Or train a single model:
  from train import train_model
  from model import ConvNextModel
  model, kappa = train_model('convnext', ConvNextModel)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from pathlib import Path

from config import (
    EPOCHS, BATCH_SIZE, LR,
    BACKBONE_LR, HEAD_LR,
    DR_LOSS_WEIGHT, EDEMA_LOSS_WEIGHT,
    MODEL_DIR,
)
from dataset import prepare_data
from model import ConvNextModel, MultimodalModel


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

torch.backends.cudnn.benchmark = True   # speeds up training on fixed input size


def get_class_weights(train_df, device):
    """Inverse frequency class weights — penalises rare class errors more."""
    dr_counts    = train_df['level'].value_counts().sort_index()
    edema_counts = train_df['edema'].value_counts().sort_index()

    dr_weights    = torch.tensor(
        (1.0 / dr_counts).values, dtype=torch.float32
    ).to(device)
    edema_weights = torch.tensor(
        (1.0 / edema_counts).values, dtype=torch.float32
    ).to(device)

    print(f'DR weights    : {dr_weights}')
    print(f'Edema weights : {edema_weights}')
    return dr_weights, edema_weights


# ─────────────────────────────────────────────────────────────────────────────
# One epoch: train
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, dr_criterion, edema_criterion, scaler):
    model.train()
    total_loss = 0.0

    for images, text_features, labels, edema_labels in tqdm(loader, leave=False):
        images       = images.to(device, non_blocking=True)
        text_features = text_features.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)
        edema_labels = edema_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast('cuda'):
            dr_out, edema_out = model(images, text_features)
            loss = (DR_LOSS_WEIGHT    * dr_criterion(dr_out, labels) +
                    EDEMA_LOSS_WEIGHT * edema_criterion(edema_out, edema_labels))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch: validate
# ─────────────────────────────────────────────────────────────────────────────

def validate_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, text_features, labels, _ in tqdm(loader, leave=False):
            images        = images.to(device, non_blocking=True)
            text_features = text_features.to(device, non_blocking=True)

            with autocast('cuda'):
                dr_out, _ = model(images, text_features)

            preds = torch.argmax(dr_out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        cohen_kappa_score(all_labels, all_preds, weights='quadratic'),
        all_preds,
        all_labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Train a single model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    mode        : str,
    ModelClass,
    train_loader,
    val_loader,
    dr_weights,
    edema_weights,
) -> tuple[float, list, list]:
    """
    Trains one model for EPOCHS and saves checkpoint.

    Args:
        mode         : 'convnext' or 'image_only'
        ModelClass   : ConvNextModel or MultimodalModel
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        dr_weights   : class weights for DR loss
        edema_weights: class weights for edema loss

    Returns:
        (best_kappa, best_preds, best_labels)
    """
    print(f'\n{"="*50}')
    print(f'  Training: {mode}')
    print(f'{"="*50}')

    model = ModelClass().to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if mode == 'image_only':
        # Freeze most of backbone, unfreeze last block
        for param in model.image_model.parameters():
            param.requires_grad = False
        for param in model.image_model.features[6:].parameters():
            param.requires_grad = True

        optimizer = optim.Adam([
            {'params': model.image_model.features[6:].parameters(), 'lr': BACKBONE_LR},
            {'params': model.text_branch.parameters(),              'lr': HEAD_LR},
            {'params': model.fusion.parameters(),                   'lr': HEAD_LR},
            {'params': model.dr_head.parameters(),                  'lr': HEAD_LR},
            {'params': model.edema_head.parameters(),               'lr': HEAD_LR},
        ])
    else:
        # ConvNext — train everything with single LR
        optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── Loss functions ────────────────────────────────────────────────────────
    dr_criterion    = nn.CrossEntropyLoss(weight=dr_weights)
    edema_criterion = nn.CrossEntropyLoss(weight=edema_weights)
    scaler          = GradScaler('cuda')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_kappa  = -1.0
    best_preds  = []
    best_labels = []

    for epoch in range(1, EPOCHS + 1):
        loss              = train_epoch(model, train_loader, optimizer,
                                        dr_criterion, edema_criterion, scaler)
        kappa, preds, labels = validate_epoch(model, val_loader)

        print(f'Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | Kappa: {kappa:.4f}')

        if kappa > best_kappa:
            best_kappa  = kappa
            best_preds  = preds
            best_labels = labels
            no_improve  = 0                        # ← add this
            ckpt_path   = MODEL_DIR / f'dr_model_{mode}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f'  ✅ Saved → {ckpt_path.name}')
        else:
            no_improve += 1                        # ← add this
            print(f'  No improvement ({no_improve}/3)')
            if no_improve >= 3:                    # ← add this
                print(f'  ⏹  Early stopping')
                break

    return best_kappa, best_preds, best_labels


# ─────────────────────────────────────────────────────────────────────────────
# Run all models and compare
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    """
    Trains both models sequentially and prints final comparison.
    Reproduces the notebook's results exactly.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    print('Preparing data...\n')
    train_dataset, val_dataset, train_loader, val_loader = prepare_data()
    train_dataset.use_captions = False
    val_dataset.use_captions   = False

    # ── Class weights ─────────────────────────────────────────────────────────
    dr_weights, edema_weights = get_class_weights(
        train_dataset.df, device
    )

    # ── Train both models ─────────────────────────────────────────────────────
    results = {}
    configs = {
        'image_only': MultimodalModel,
        'convnext'  : ConvNextModel,
    }

    for mode, ModelClass in configs.items():
        kappa, preds, labels = train_model(
            mode, ModelClass,
            train_loader, val_loader,
            dr_weights, edema_weights,
        )
        results[mode] = {'kappa': kappa, 'preds': preds, 'labels': labels}

    # ── Final comparison ──────────────────────────────────────────────────────
    print(f'\n{"="*50}')
    print('  FINAL COMPARISON')
    print(f'{"="*50}')
    for mode, r in results.items():
        print(f'  {mode:<15} Kappa = {r["kappa"]:.4f}')
    print(f'{"="*50}')

    return results


if __name__ == '__main__':
    run_all()
