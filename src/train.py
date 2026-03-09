"""
train.py
────────
Full two-phase training loop for the DR classifier.

Phase 1 : Frozen backbone  — train head only  (~10 epochs, fast)
Phase 2 : Unfrozen backbone — fine-tune all   (~20 epochs, slow)

Saves:
  models/best_model.pth        ← best validation kappa checkpoint
  results/training_curves.png  ← loss + kappa curves for both phases

Usage:
  python src/train.py
  python src/train.py --epochs1 10 --epochs2 20 --batch_size 32
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

# ── Make src/ importable when running as a script ────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))

from dataset_loader import get_dataloaders
from model import build_model, get_optimizer, count_parameters


# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
WEIGHTS_DIR = BASE_DIR / 'results'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    return device


def load_class_weights(device: torch.device) -> torch.Tensor | None:
    """Load precomputed class weights from results/class_weights.json."""
    path = RESULTS_DIR / 'class_weights.json'
    if not path.exists():
        print('⚠️  class_weights.json not found — using unweighted loss')
        return None
    with open(path) as f:
        w = json.load(f)
    weights = torch.tensor([w[str(i)] for i in range(5)], dtype=torch.float32)
    weights = weights.to(device)
    print(f'Class weights loaded: {[f"{x:.3f}" for x in weights.cpu().tolist()]}')
    return weights


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Quadratic Weighted Kappa — the standard DR evaluation metric."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# ─────────────────────────────────────────────────────────────────────────────
# One epoch: train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, total_epochs
) -> tuple[float, float]:
    """
    Runs one full training epoch.
    Returns (avg_loss, kappa).
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'  Train [{epoch}/{total_epochs}]', leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in phase 2
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(loader.dataset)
    kappa    = quadratic_weighted_kappa(all_labels, all_preds)
    return avg_loss, kappa


# ─────────────────────────────────────────────────────────────────────────────
# One epoch: validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model, loader, criterion, device, epoch, total_epochs
) -> tuple[float, float]:
    """
    Runs one full validation epoch.
    Returns (avg_loss, kappa).
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'  Val   [{epoch}/{total_epochs}]', leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    kappa    = quadratic_weighted_kappa(all_labels, all_preds)
    return avg_loss, kappa


# ─────────────────────────────────────────────────────────────────────────────
# Training phase
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(
    phase        : int,
    model        : nn.Module,
    train_loader,
    val_loader,
    criterion    : nn.Module,
    device       : torch.device,
    num_epochs   : int,
    history      : dict,
    best_kappa   : float,
    patience     : int = 5,
) -> tuple[float, dict]:
    """
    Runs a full training phase (1 or 2).

    Returns:
        (best_kappa_so_far, updated_history)
    """

    optimizer = get_optimizer(model, phase=phase)

    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    no_improve = 0   # early stopping counter

    print(f'\n{"─"*55}')
    print(f'  PHASE {phase}  |  {num_epochs} epochs  |  '
          f'{"Backbone FROZEN" if phase == 1 else "Backbone UNFROZEN"}')
    print(f'{"─"*55}')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_kappa = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        val_loss, val_kappa = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[-1]['lr']

        # ── Log ───────────────────────────────────────────────────────────────
        print(
            f'  Ep {epoch:>3}/{num_epochs} | '
            f'Loss {train_loss:.4f}/{val_loss:.4f} | '
            f'Kappa {train_kappa:.4f}/{val_kappa:.4f} | '
            f'LR {lr_now:.2e} | '
            f'{elapsed:.0f}s'
        )

        history[f'p{phase}_train_loss'].append(train_loss)
        history[f'p{phase}_val_loss'].append(val_loss)
        history[f'p{phase}_train_kappa'].append(train_kappa)
        history[f'p{phase}_val_kappa'].append(val_kappa)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            no_improve = 0
            ckpt_path  = MODELS_DIR / 'best_model.pth'
            torch.save({
                'epoch'      : epoch,
                'phase'      : phase,
                'model_state': model.state_dict(),
                'val_kappa'  : val_kappa,
                'val_loss'   : val_loss,
            }, ckpt_path)
            print(f'  ✅ New best kappa {best_kappa:.4f} → saved to {ckpt_path}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  ⏹  Early stopping (no improvement for {patience} epochs)')
                break

    return best_kappa, history


# ─────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """Saves loss and kappa curves to results/training_curves.png."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — Diabetic Retinopathy Classifier',
                 fontsize=14, fontweight='bold')

    colors = {'p1': ('#2196F3', '#F44336'), 'p2': ('#4CAF50', '#FF9800')}

    for phase, label in [(1, 'Phase 1'), (2, 'Phase 2')]:
        key = f'p{phase}'
        c_train, c_val = colors[key]

        tl = history.get(f'{key}_train_loss', [])
        vl = history.get(f'{key}_val_loss',   [])
        tk = history.get(f'{key}_train_kappa', [])
        vk = history.get(f'{key}_val_kappa',   [])

        if not tl:
            continue

        x = range(1, len(tl) + 1)

        # Loss
        axes[0].plot(x, tl, color=c_train, linestyle='--',
                     label=f'{label} Train', linewidth=1.5)
        axes[0].plot(x, vl, color=c_val,
                     label=f'{label} Val', linewidth=2)

        # Kappa
        axes[1].plot(x, tk, color=c_train, linestyle='--',
                     label=f'{label} Train', linewidth=1.5)
        axes[1].plot(x, vk, color=c_val,
                     label=f'{label} Val', linewidth=2)

    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Quadratic Weighted Kappa')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('QWK Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nTraining curves saved → {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    device = get_device()
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print('Loading data...')
    train_loader, val_loader, _ = get_dataloaders(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print('\nBuilding model...')
    model = build_model(num_classes=5, dropout=args.dropout, freeze=True)
    model = model.to(device)
    params = count_parameters(model)
    print(f'Trainable params : {params["trainable"]:,}')
    print(f'Total params     : {params["total"]:,}')

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_weights = load_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        'p1_train_loss' : [], 'p1_val_loss' : [],
        'p1_train_kappa': [], 'p1_val_kappa': [],
        'p2_train_loss' : [], 'p2_val_loss' : [],
        'p2_train_kappa': [], 'p2_val_kappa': [],
    }

    best_kappa = -1.0

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    best_kappa, history = run_phase(
        phase        = 1,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        device       = device,
        num_epochs   = args.epochs1,
        history      = history,
        best_kappa   = best_kappa,
        patience     = args.patience,
    )

    # ── Phase 2: unfreeze and fine-tune ───────────────────────────────────────
    model.unfreeze_backbone()

    best_kappa, history = run_phase(
        phase        = 2,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        device       = device,
        num_epochs   = args.epochs2,
        history      = history,
        best_kappa   = best_kappa,
        patience     = args.patience,
    )

    # ── Save curves ───────────────────────────────────────────────────────────
    plot_training_curves(history)

    print(f'\n{"═"*55}')
    print(f'  TRAINING COMPLETE')
    print(f'  Best Val Kappa : {best_kappa:.4f}')
    print(f'  Model saved    : {MODELS_DIR / "best_model.pth"}')
    print(f'{"═"*55}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train DR classifier')
    p.add_argument('--epochs1',     type=int,   default=10,
                   help='Phase 1 epochs — frozen backbone (default: 10)')
    p.add_argument('--epochs2',     type=int,   default=20,
                   help='Phase 2 epochs — fine-tune (default: 20)')
    p.add_argument('--batch_size',  type=int,   default=32,
                   help='Batch size (default: 32, reduce to 16 if OOM)')
    p.add_argument('--dropout',     type=float, default=0.4,
                   help='Head dropout rate (default: 0.4)')
    p.add_argument('--num_workers', type=int,   default=0,
                   help='Dataloader workers (default: 0, safe for Windows)')
    p.add_argument('--patience',    type=int,   default=5,
                   help='Early stopping patience (default: 5)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)