"""
train.py
────────
Train a single model. Run twice — once per architecture.

Step 1: python src/train.py --arch efficientnet_b4 --save_name effb4_best.pth
Step 2: python src/train.py --arch resnet50         --save_name resnet50_best.pth
Step 3: python src/evaluate.py  (ensemble both)
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


BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
    return device


def load_class_weights(device):
    path = RESULTS_DIR / 'class_weights.json'
    if not path.exists():
        return None
    with open(path) as f:
        w = json.load(f)
    weights = torch.tensor([w[str(i)] for i in range(5)], dtype=torch.float32).to(device)
    print(f'Class weights: {[f"{x:.3f}" for x in weights.cpu().tolist()]}')
    return weights


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f'  Train [{epoch}/{total}]', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        all_preds.extend(model(images).detach().argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return running_loss / len(loader.dataset), qwk(all_labels, all_preds)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc=f'  Val   [{epoch}/{total}]', leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader.dataset), qwk(all_labels, all_preds)


def run_phase(phase, model, train_loader, val_loader, criterion,
              device, num_epochs, history, best_kappa, patience, arch, save_name):

    optimizer = get_optimizer(model, phase=phase, arch=arch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-7, verbose=True
    )
    no_improve = 0

    print(f'\n{"─"*60}')
    print(f'  PHASE {phase}  [{arch}]  |  {num_epochs} epochs  |  '
          f'{"FROZEN" if phase == 1 else "UNFROZEN"}')
    print(f'{"─"*60}')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_kappa = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_kappa = validate(
            model, val_loader, criterion, device, epoch, num_epochs)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[-1]['lr']
        print(f'  Ep {epoch:>3}/{num_epochs} | '
              f'Loss {train_loss:.4f}/{val_loss:.4f} | '
              f'Kappa {train_kappa:.4f}/{val_kappa:.4f} | '
              f'LR {lr_now:.2e} | {time.time()-t0:.0f}s')

        history[f'p{phase}_train_loss'].append(train_loss)
        history[f'p{phase}_val_loss'].append(val_loss)
        history[f'p{phase}_train_kappa'].append(train_kappa)
        history[f'p{phase}_val_kappa'].append(val_kappa)

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            no_improve = 0
            ckpt = MODELS_DIR / save_name
            torch.save({
                'epoch': epoch, 'phase': phase,
                'arch': arch,
                'model_state': model.state_dict(),
                'val_kappa': val_kappa, 'val_loss': val_loss,
            }, ckpt)
            print(f'  ✅ Best kappa {best_kappa:.4f} → {ckpt.name}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  ⏹  Early stopping (patience={patience})')
                break

    return best_kappa, history


def plot_curves(history, arch):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {arch}', fontsize=14, fontweight='bold')
    colors = {'p1': ('#2196F3', '#F44336'), 'p2': ('#4CAF50', '#FF9800')}
    for phase in [1, 2]:
        key = f'p{phase}'
        ct, cv = colors[key]
        tl = history.get(f'{key}_train_loss', [])
        vl = history.get(f'{key}_val_loss', [])
        tk = history.get(f'{key}_train_kappa', [])
        vk = history.get(f'{key}_val_kappa', [])
        if not tl:
            continue
        x = range(1, len(tl) + 1)
        axes[0].plot(x, tl, color=ct, linestyle='--', label=f'P{phase} Train', linewidth=1.5)
        axes[0].plot(x, vl, color=cv, label=f'P{phase} Val', linewidth=2)
        axes[1].plot(x, tk, color=ct, linestyle='--', label=f'P{phase} Train', linewidth=1.5)
        axes[1].plot(x, vk, color=cv, label=f'P{phase} Val', linewidth=2)
    for ax, title, ylabel in zip(axes, ['Loss', 'QWK'], ['Loss', 'Kappa']):
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = RESULTS_DIR / f'training_curves_{arch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Curves saved → {save_path}')
    plt.close()


def train(args):
    device = get_device()
    print(f'\nArchitecture : {args.arch}')
    print(f'Save name    : {args.save_name}\n')

    print('Loading data...')
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=0, oversample=True)

    print('\nBuilding model...')
    model  = build_model(arch=args.arch, dropout=0.4, freeze=True).to(device)
    params = count_parameters(model)
    print(f'Trainable : {params["trainable"]:,} | Total : {params["total"]:,}')

    class_weights = load_class_weights(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    history = {k: [] for k in [
        'p1_train_loss','p1_val_loss','p1_train_kappa','p1_val_kappa',
        'p2_train_loss','p2_val_loss','p2_train_kappa','p2_val_kappa',
    ]}
    best_kappa = -1.0

    # Phase 1
    best_kappa, history = run_phase(
        1, model, train_loader, val_loader, criterion, device,
        args.epochs1, history, best_kappa, args.patience, args.arch, args.save_name)

    # Phase 2
    model.unfreeze_backbone()
    best_kappa, history = run_phase(
        2, model, train_loader, val_loader, criterion, device,
        args.epochs2, history, best_kappa, args.patience, args.arch, args.save_name)

    plot_curves(history, args.arch)

    print(f'\n{"═"*55}')
    print(f'  DONE  [{args.arch}]')
    print(f'  Best Val Kappa : {best_kappa:.4f}')
    print(f'  Checkpoint     : models/{args.save_name}')
    print(f'{"═"*55}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arch',        type=str,   default='efficientnet_b4',
                   choices=['efficientnet_b4', 'resnet50'])
    p.add_argument('--save_name',   type=str,   default='best_model.pth')
    p.add_argument('--epochs1',     type=int,   default=10)
    p.add_argument('--epochs2',     type=int,   default=30)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--patience',    type=int,   default=7)
    args = p.parse_args()
    train(args)
