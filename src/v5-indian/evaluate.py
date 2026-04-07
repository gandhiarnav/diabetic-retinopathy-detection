"""
evaluate.py
───────────
Loads a saved model and evaluates it on the validation set.

Usage:
  from evaluate import evaluate
  evaluate('convnext')      # loads dr_model_convnext.pth
  evaluate('image_only')    # loads dr_model_image_only.pth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
)

from config import MODEL_DIR, RESULTS_DIR, NUM_DR_CLASSES
from dataset import prepare_data
from model import ConvNextModel, MultimodalModel
from train import validate_epoch, device

GRADE_LABELS = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}


def evaluate(mode: str = 'convnext'):
    """
    Loads saved checkpoint and runs full evaluation.

    Args:
        mode : 'convnext' or 'image_only'
    """

    # ── Load model ────────────────────────────────────────────────────────────
    ModelClass = ConvNextModel if mode == 'convnext' else MultimodalModel
    model      = ModelClass().to(device)
    ckpt_path  = MODEL_DIR / f'dr_model_{mode}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded: {ckpt_path.name}\n')

    # ── Data ──────────────────────────────────────────────────────────────────
    _, val_dataset, _, val_loader = prepare_data()
    val_dataset.use_captions = False

    # ── Inference ─────────────────────────────────────────────────────────────
    kappa, preds, labels = validate_epoch(model, val_loader)
    preds  = np.array(preds)
    labels = np.array(labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(labels, preds)

    sep = '═' * 52
    print(sep)
    print(f'  EVALUATION — {mode}')
    print(sep)
    print(f'  Accuracy               : {acc*100:.2f}%')
    print(f'  Quadratic Weighted Kappa: {kappa:.4f}')
    print()
    print(classification_report(
        labels, preds,
        target_names=[GRADE_LABELS[i] for i in range(5)],
        zero_division=0,
    ))
    print(sep)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm     = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    xlabels = [f'Grade {i}\n{GRADE_LABELS[i]}' for i in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Confusion Matrix — {mode} (Kappa: {kappa:.4f})',
                 fontsize=14, fontweight='bold')

    sns.heatmap(cm,     annot=True, fmt='d',   cmap='Blues',
                xticklabels=xlabels, yticklabels=xlabels, ax=axes[0])
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=xlabels, yticklabels=xlabels, ax=axes[1])
    axes[1].set_title('Row-Normalised (%)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    plt.tight_layout()
    save_path = RESULTS_DIR / f'confusion_matrix_{mode}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Confusion matrix → {save_path}')
    plt.show()
