"""
evaluate.py
───────────
Loads the best saved model and evaluates it on the held-out test set.

Metrics:
  - Accuracy
  - Quadratic Weighted Kappa (QWK)
  - Precision, Recall, F1  (per class + weighted average)
  - Confusion Matrix

Saves:
  results/confusion_matrix.png
  results/evaluation_report.txt

Usage:
  python src/evaluate.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))

from dataset_loader import get_dataloaders
from model import build_model


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative',
}


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> nn.Module:
    ckpt_path = MODELS_DIR / 'best_model.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'No checkpoint found at {ckpt_path}\n'
            f'Run train.py first to generate best_model.pth'
        )

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Build model with backbone unfrozen (Phase 2 weights)
    model = build_model(num_classes=5, dropout=0.4, freeze=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    print(f'Loaded checkpoint from : {ckpt_path}')
    print(f'  Saved at epoch       : {checkpoint["epoch"]}')
    print(f'  Phase                : {checkpoint["phase"]}')
    print(f'  Val Kappa at save    : {checkpoint["val_kappa"]:.4f}')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Run inference on test set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model, test_loader, device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs model on the full test set.
    Returns (all_labels, all_preds) as numpy arrays.
    """
    all_labels, all_preds = [], []

    for images, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    labels = [f'Grade {i}\n{GRADE_LABELS[i]}' for i in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix — DR Classifier (Test Set)',
                 fontsize=14, fontweight='bold')

    # ── Raw counts ────────────────────────────────────────────────────────────
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=axes[0]
    )
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # ── Percentages ───────────────────────────────────────────────────────────
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Oranges',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=axes[1]
    )
    axes[1].set_title('Row-Normalised (%)')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    save_path = RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nConfusion matrix saved → {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Compute and print all metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels     = list(range(5))
    label_names = [GRADE_LABELS[i] for i in labels]

    accuracy  = accuracy_score(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    precision = precision_score(y_true, y_pred, average='weighted',
                                zero_division=0, labels=labels)
    recall    = recall_score(y_true, y_pred, average='weighted',
                             zero_division=0, labels=labels)
    f1        = f1_score(y_true, y_pred, average='weighted',
                         zero_division=0, labels=labels)

    # Per-class scores
    per_class_p  = precision_score(y_true, y_pred, average=None,
                                   zero_division=0, labels=labels)
    per_class_r  = recall_score(y_true, y_pred, average=None,
                                zero_division=0, labels=labels)
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                            zero_division=0, labels=labels)

    metrics = {
        'accuracy'          : accuracy,
        'quadratic_kappa'   : kappa,
        'weighted_precision': precision,
        'weighted_recall'   : recall,
        'weighted_f1'       : f1,
        'per_class'         : {
            GRADE_LABELS[i]: {
                'precision': per_class_p[i],
                'recall'   : per_class_r[i],
                'f1'       : per_class_f1[i],
            }
            for i in labels
        }
    }

    return metrics


def print_report(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    sep  = '═' * 52
    sep2 = '─' * 52

    lines = [
        sep,
        '  EVALUATION REPORT — DR Classifier (Test Set)',
        sep,
        f'  Accuracy               : {metrics["accuracy"]*100:.2f}%',
        f'  Quadratic Weighted Kappa: {metrics["quadratic_kappa"]:.4f}',
        f'  Weighted Precision     : {metrics["weighted_precision"]:.4f}',
        f'  Weighted Recall        : {metrics["weighted_recall"]:.4f}',
        f'  Weighted F1            : {metrics["weighted_f1"]:.4f}',
        sep2,
        f'  {"Class":<20} {"Precision":>10} {"Recall":>10} {"F1":>8}',
        sep2,
    ]

    for grade, scores in metrics['per_class'].items():
        lines.append(
            f'  {grade:<20} '
            f'{scores["precision"]:>10.4f} '
            f'{scores["recall"]:>10.4f} '
            f'{scores["f1"]:>8.4f}'
        )

    lines += [
        sep2,
        '',
        '  Full sklearn report:',
        sep2,
        classification_report(
            y_true, y_pred,
            target_names=[GRADE_LABELS[i] for i in range(5)],
            zero_division=0
        ),
        sep,
        '  Kappa interpretation:',
        '    < 0.20  Poor    |  0.20-0.40  Fair',
        '    0.40-0.60  Moderate  |  0.60-0.80  Good',
        '    > 0.80  Excellent',
        sep,
    ]

    report = '\n'.join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(device)

    # ── Load test set only ────────────────────────────────────────────────────
    print('\nLoading test data...')
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=0)

    # ── Predict ───────────────────────────────────────────────────────────────
    print()
    y_true, y_pred = get_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print()
    metrics = compute_metrics(y_true, y_pred)
    report  = print_report(metrics, y_true, y_pred)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = RESULTS_DIR / 'evaluation_report.txt'
    report_path.write_text(report)
    print(f'Evaluation report saved → {report_path}')

    # ── Save metrics as JSON ──────────────────────────────────────────────────
    json_path = RESULTS_DIR / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics JSON saved      → {json_path}')


if __name__ == '__main__':
    evaluate()