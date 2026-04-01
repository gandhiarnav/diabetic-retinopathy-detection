"""
train.py
────────
Two-phase training loop for the DR classifier.

Improvements over v1:
  - ReduceLROnPlateau scheduler (replaces cosine — more responsive)
  - Loads oversampled balanced dataset automatically
  - Works with combined APTOS + IDRiD dataset

Phase 1 : Frozen backbone  — train head only  (~10 epochs)
Phase 2 : Unfrozen backbone — fine-tune all   (~30 epochs)

Usage:
  python src/train.py
  python src/train.py --epochs1 10 --epochs2 30 --batch_size 16
"""

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

sys.path.append(str(Path(__file__).resolve().parent))

from dataset_loader import get_dataloaders
from model import build_model, get_optimizer, count_parameters


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
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
    path = RESULTS_DIR / 'class_weights.json'
    if not path.exists():
        print('⚠️  class_weights.json not found — using unweighted loss')
        return None
    with open(path) as f:
        w = json.load(f)
    weights = torch.tensor([w[str(i)] for i in range(5)], dtype=torch.float32).to(device)
    print(f'Class weights: {[f"{x:.3f}" for x in weights.cpu().tolist()]}')
    return weights


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f'  Train [{epoch}/{total}]', leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(loader.dataset)
    kappa    = quadratic_weighted_kappa(all_labels, all_preds)
    return avg_loss, kappa


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f'  Val   [{epoch}/{total}]', leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    kappa    = quadratic_weighted_kappa(all_labels, all_preds)
    return avg_loss, kappa


# ─────────────────────────────────────────────────────────────────────────────
# Training phase
# ─────────────────────────────────────────────────────────────────────────────

def run_phase(
    phase, model, train_loader, val_loader,
    criterion, device, num_epochs, history,
    best_kappa, patience=7,
):
    optimizer = get_optimizer(model, phase=phase)

    # ── ReduceLROnPlateau — reduces LR when val_loss stops improving ──────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # monitor val_loss (lower is better)
        factor=0.3,       # multiply LR by 0.3 on plateau
        patience=3,       # wait 3 epochs before reducing
        min_lr=1e-7,
        verbose=True,
    )

    no_improve = 0

    print(f'\n{"─"*58}')
    print(f'  PHASE {phase}  |  {num_epochs} epochs  |  '
          f'{"Backbone FROZEN" if phase == 1 else "Backbone UNFROZEN"}')
    print(f'{"─"*58}')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_kappa = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        val_loss, val_kappa = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        # ReduceLROnPlateau steps on val_loss
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[-1]['lr']

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

        # ── Checkpoint on best val kappa ──────────────────────────────────────
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
            print(f'  ✅ New best kappa {best_kappa:.4f} → saved')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  ⏹  Early stopping triggered (patience={patience})')
                break

    return best_kappa, history


# ─────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — DR Classifier', fontsize=14, fontweight='bold')

    colors = {'p1': ('#2196F3', '#F44336'), 'p2': ('#4CAF50', '#FF9800')}

    for phase in [1, 2]:
        key = f'p{phase}'
        c_train, c_val = colors[key]
        tl = history.get(f'{key}_train_loss', [])
        vl = history.get(f'{key}_val_loss',   [])
        tk = history.get(f'{key}_train_kappa', [])
        vk = history.get(f'{key}_val_kappa',   [])
        if not tl:
            continue
        x = range(1, len(tl) + 1)
        axes[0].plot(x, tl, color=c_train, linestyle='--',
                     label=f'Phase {phase} Train', linewidth=1.5)
        axes[0].plot(x, vl, color=c_val,
                     label=f'Phase {phase} Val', linewidth=2)
        axes[1].plot(x, tk, color=c_train, linestyle='--',
                     label=f'Phase {phase} Train', linewidth=1.5)
        axes[1].plot(x, vk, color=c_val,
                     label=f'Phase {phase} Val', linewidth=2)

    for ax, title, ylabel in zip(axes,
        ['Loss', 'Quadratic Weighted Kappa'],
        ['Cross-Entropy Loss', 'QWK Score']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

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
        oversample  = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print('\nBuilding model...')
    model  = build_model(num_classes=5, dropout=0.4, freeze=True)
    model  = model.to(device)
    params = count_parameters(model)
    print(f'Trainable : {params["trainable"]:,}')
    print(f'Total     : {params["total"]:,}')

    # ── Loss ──────────────────────────────────────────────────────────────────
    # With oversampling, classes are balanced — unweighted loss is fine.
    # Class weights still help slightly with remaining noise.
    class_weights = load_class_weights(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        'p1_train_loss': [], 'p1_val_loss': [],
        'p1_train_kappa': [], 'p1_val_kappa': [],
        'p2_train_loss': [], 'p2_val_loss': [],
        'p2_train_kappa': [], 'p2_val_kappa': [],
    }
    best_kappa = -1.0

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    best_kappa, history = run_phase(
        phase=1, model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device,
        num_epochs=args.epochs1, history=history,
        best_kappa=best_kappa, patience=args.patience,
    )

    # ── Phase 2: fine-tune ────────────────────────────────────────────────────
    model.unfreeze_backbone()
    best_kappa, history = run_phase(
        phase=2, model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device,
        num_epochs=args.epochs2, history=history,
        best_kappa=best_kappa, patience=args.patience,
    )

    plot_training_curves(history)

    print(f'\n{"═"*55}')
    print(f'  TRAINING COMPLETE')
    print(f'  Best Val Kappa : {best_kappa:.4f}')
    print(f'  Model saved    : {MODELS_DIR / "best_model.pth"}')
    print(f'{"═"*55}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train DR classifier')
    p.add_argument('--epochs1',     type=int,   default=10)
    p.add_argument('--epochs2',     type=int,   default=30)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--dropout',     type=float, default=0.4)
    p.add_argument('--num_workers', type=int,   default=0)
    p.add_argument('--patience',    type=int,   default=7)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
