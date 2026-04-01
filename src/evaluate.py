"""
evaluate.py
───────────
Evaluates a single model OR an ensemble of models on the test set.

Single model:
  python src/evaluate.py --models effb4_best.pth

Ensemble (average predictions):
  python src/evaluate.py --models effb4_best.pth resnet50_best.pth
"""

import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(str(Path(__file__).resolve().parent))
from dataset_loader import get_dataloaders
from model import build_model

BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRADE_LABELS = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_name: str, device: torch.device):
    ckpt_path  = MODELS_DIR / ckpt_name
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    arch  = checkpoint.get('arch', 'efficientnet_b4')
    model = build_model(arch=arch, freeze=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device).eval()

    print(f'  Loaded : {ckpt_name}')
    print(f'  Arch   : {arch}')
    print(f'  Epoch  : {checkpoint["epoch"]} (Phase {checkpoint["phase"]})')
    print(f'  Val κ  : {checkpoint["val_kappa"]:.4f}')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Get softmax probabilities from one model
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, probabilities) — probs shape: (N, 5)"""
    all_probs, all_labels = [], []
    softmax = torch.nn.Softmax(dim=1)

    for images, labels in tqdm(loader, desc=f'  Inferring', leave=False):
        images  = images.to(device, non_blocking=True)
        outputs = softmax(model(images))
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + plots
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        'accuracy'          : accuracy_score(y_true, y_pred),
        'quadratic_kappa'   : cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall'   : recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1'       : f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title_suffix=''):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = [f'Grade {i}\n{GRADE_LABELS[i]}' for i in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Confusion Matrix {title_suffix}', fontsize=14, fontweight='bold')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Row-Normalised (%)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    plt.tight_layout()
    fname = 'confusion_matrix_ensemble.png' if 'Ensemble' in title_suffix else 'confusion_matrix.png'
    save_path = RESULTS_DIR / fname
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Confusion matrix → {save_path}')
    plt.close()


def print_report(y_true, y_pred, metrics, title):
    sep  = '═' * 55
    sep2 = '─' * 55
    lines = [
        sep,
        f'  {title}',
        sep,
        f'  Accuracy               : {metrics["accuracy"]*100:.2f}%',
        f'  Quadratic Weighted Kappa: {metrics["quadratic_kappa"]:.4f}',
        f'  Weighted Precision      : {metrics["weighted_precision"]:.4f}',
        f'  Weighted Recall         : {metrics["weighted_recall"]:.4f}',
        f'  Weighted F1             : {metrics["weighted_f1"]:.4f}',
        sep2,
        classification_report(y_true, y_pred,
            target_names=[GRADE_LABELS[i] for i in range(5)], zero_division=0),
        sep,
        '  Kappa: <0.4 Poor | 0.4-0.6 Moderate | 0.6-0.8 Good | >0.8 Excellent',
        sep,
    ]
    report = '\n'.join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    # ── Load models ───────────────────────────────────────────────────────────
    print(f'Loading {len(args.models)} model(s)...')
    models = [load_model(name, device) for name in args.models]
    print()

    # ── Test data ─────────────────────────────────────────────────────────────
    print('Loading test data...')
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=0, oversample=False)
    print()

    # ── Get probabilities from each model ─────────────────────────────────────
    all_probs = []
    y_true    = None

    for i, (model, name) in enumerate(zip(models, args.models)):
        print(f'Running inference — {name}')
        labels, probs = get_probabilities(model, test_loader, device)
        all_probs.append(probs)
        if y_true is None:
            y_true = labels

        # Single model metrics
        y_pred_single  = probs.argmax(axis=1)
        metrics_single = compute_metrics(y_true, y_pred_single)
        print(f'  Kappa : {metrics_single["quadratic_kappa"]:.4f}  |  '
              f'Accuracy : {metrics_single["accuracy"]*100:.2f}%')
        print()

    # ── Ensemble: average probabilities ───────────────────────────────────────
    avg_probs  = np.mean(all_probs, axis=0)   # shape: (N, 5)
    y_pred     = avg_probs.argmax(axis=1)
    metrics    = compute_metrics(y_true, y_pred)

    # ── Report ────────────────────────────────────────────────────────────────
    title = ('ENSEMBLE EVALUATION' if len(args.models) > 1
             else f'EVALUATION — {args.models[0]}')
    report = print_report(y_true, y_pred, metrics, title)
    plot_confusion_matrix(y_true, y_pred,
                          title_suffix='— Ensemble' if len(args.models) > 1 else '')

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix = '_ensemble' if len(args.models) > 1 else ''
    report_path = RESULTS_DIR / f'evaluation_report{suffix}.txt'
    report_path.write_text(report)

    metrics_out = {**metrics, 'models': args.models}
    with open(RESULTS_DIR / f'metrics{suffix}.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    print(f'Report saved → {report_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['best_model.pth'],
                   help='One or more checkpoint filenames from models/ folder')
    args = p.parse_args()
    evaluate(args)
