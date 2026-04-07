"""
evaluate.py
───────────
Full evaluation of trained DR models with rich visualisations.

Generates:
  Text output  : individual reports + side-by-side comparison
  Charts saved to results/:
    01_confusion_matrix_counts.png      raw count heatmaps (both models)
    02_confusion_matrix_pct.png         normalised % heatmaps (both models)
    03_per_class_f1.png                 F1 per grade, both models grouped
    04_per_class_precision_recall.png   precision + recall bars
    05_kappa_accuracy_comparison.png    overall metrics bar chart
    06_roc_curves.png                   one-vs-rest ROC per grade
    07_prediction_distribution.png      what each model predicts most
    08_error_analysis.png               where each model fails

Usage:
  from evaluate import evaluate_all
  evaluate_all()                        # evaluates both models
  evaluate_all(modes=['convnext'])      # single model only
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from config import MODEL_DIR, RESULTS_DIR, NUM_DR_CLASSES
from dataset import prepare_data
from model import ConvNextModel, MultimodalModel
from train import validate_epoch, device

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.dpi'        : 130,
})

GRADE_LABELS  = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
GRADE_COLORS  = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
MODEL_COLORS  = {'image_only': '#2196F3', 'convnext': '#F44336'}
MODEL_NAMES   = {'image_only': 'EfficientNetB0', 'convnext': 'ConvNeXt-Tiny'}


# ─────────────────────────────────────────────────────────────────────────────
# Load + run inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(mode: str):
    ModelClass = ConvNextModel if mode == 'convnext' else MultimodalModel
    model      = ModelClass().to(device)
    ckpt_path  = MODEL_DIR / f'dr_model_{mode}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'No checkpoint at {ckpt_path} — run train.py first')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_all_results(modes, val_loader, val_dataset) -> dict:
    """Runs inference for each model, returns dict of results."""
    all_results = {}
    for mode in modes:
        print(f'  Running inference — {MODEL_NAMES[mode]}...')
        model              = load_model(mode)
        val_dataset.use_captions = False
        kappa, preds, labels = validate_epoch(model, val_loader)
        all_results[mode]  = {
            'preds'  : np.array(preds),
            'labels' : np.array(labels),
            'kappa'  : kappa,
        }
        print(f'    Kappa: {kappa:.4f}  |  Acc: {accuracy_score(labels, preds)*100:.2f}%')
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Text reports
# ─────────────────────────────────────────────────────────────────────────────

def print_individual_report(mode: str, results: dict):
    r      = results[mode]
    labels = r['labels']
    preds  = r['preds']
    acc    = accuracy_score(labels, preds)
    kappa  = r['kappa']
    p_w    = precision_score(labels, preds, average='weighted', zero_division=0)
    r_w    = recall_score(labels, preds, average='weighted', zero_division=0)
    f1_w   = f1_score(labels, preds, average='weighted', zero_division=0)

    sep  = '═' * 58
    sep2 = '─' * 58
    print(f'\n{sep}')
    print(f'  MODEL: {MODEL_NAMES[mode]}')
    print(sep)
    print(f'  Accuracy                : {acc*100:.2f}%')
    print(f'  Quadratic Weighted Kappa: {kappa:.4f}')
    print(f'  Weighted Precision      : {p_w:.4f}')
    print(f'  Weighted Recall         : {r_w:.4f}')
    print(f'  Weighted F1             : {f1_w:.4f}')
    print(sep2)
    print(f'  Per-class breakdown:')
    print(sep2)
    print(f'  {"Grade":<20} {"Precision":>10} {"Recall":>8} {"F1":>8} {"Support":>9}')
    print(sep2)
    p_cls = precision_score(labels, preds, average=None, zero_division=0)
    r_cls = recall_score(labels, preds, average=None, zero_division=0)
    f_cls = f1_score(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(GRADE_LABELS):
        support = (labels == i).sum()
        print(f'  {name:<20} {p_cls[i]:>10.4f} {r_cls[i]:>8.4f} {f_cls[i]:>8.4f} {support:>9}')
    print(sep)
    print()


def print_side_by_side(results: dict):
    modes  = list(results.keys())
    sep    = '═' * 75
    sep2   = '─' * 75

    print(f'\n{sep}')
    print(f'  SIDE-BY-SIDE COMPARISON')
    print(sep)
    print(f'  {"Metric":<30} '
          + ''.join(f'{MODEL_NAMES[m]:>20}' for m in modes))
    print(sep2)

    metrics = {}
    for mode in modes:
        r = results[mode]
        metrics[mode] = {
            'Accuracy'         : accuracy_score(r['labels'], r['preds']) * 100,
            'QW Kappa'         : r['kappa'],
            'Weighted F1'      : f1_score(r['labels'], r['preds'], average='weighted', zero_division=0),
            'Weighted Precision': precision_score(r['labels'], r['preds'], average='weighted', zero_division=0),
            'Weighted Recall'  : recall_score(r['labels'], r['preds'], average='weighted', zero_division=0),
        }

    for metric in ['Accuracy', 'QW Kappa', 'Weighted F1', 'Weighted Precision', 'Weighted Recall']:
        vals = [metrics[m][metric] for m in modes]
        best = max(range(len(vals)), key=lambda i: vals[i])
        row  = f'  {metric:<30}'
        for i, (m, v) in enumerate(zip(modes, vals)):
            fmt = f'{v:.2f}%' if metric == 'Accuracy' else f'{v:.4f}'
            tag = ' ✅' if i == best and len(modes) > 1 else ''
            row += f'{fmt+tag:>20}'
        print(row)

    print(sep2)
    print(f'  Per-class F1 scores:')
    print(sep2)
    print(f'  {"Grade":<30} '
          + ''.join(f'{MODEL_NAMES[m]:>20}' for m in modes))
    print(sep2)
    for i, name in enumerate(GRADE_LABELS):
        row = f'  {name:<30}'
        f1s = [f1_score(results[m]['labels'], results[m]['preds'],
                        average=None, zero_division=0)[i] for m in modes]
        best = max(range(len(f1s)), key=lambda j: f1s[j])
        for j, f in enumerate(f1s):
            tag = ' ✅' if j == best and len(modes) > 1 else ''
            row += f'{f"{f:.4f}"+tag:>20}'
        print(row)
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 & 2 — Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict):
    modes = list(results.keys())
    n     = len(modes)

    for chart_num, (fmt, cmap, title_suffix) in enumerate([
        ('d',   'Blues',   'Raw Counts'),
        ('.1f', 'Oranges', 'Row-Normalised (%)'),
    ], start=1):
        fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
        if n == 1:
            axes = [axes]
        fig.suptitle(f'Confusion Matrix — {title_suffix}',
                     fontsize=15, fontweight='bold', y=1.01)

        for ax, mode in zip(axes, modes):
            r  = results[mode]
            cm = confusion_matrix(r['labels'], r['preds'])
            if fmt == '.1f':
                cm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

            sns.heatmap(
                cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=GRADE_LABELS,
                yticklabels=GRADE_LABELS,
                linewidths=0.5, ax=ax,
                annot_kws={'size': 11},
            )
            ax.set_title(f'{MODEL_NAMES[mode]}\nKappa: {r["kappa"]:.4f}',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('True',      fontsize=11)
            ax.tick_params(axis='x', rotation=30)
            ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        fname = f'0{chart_num}_confusion_matrix_{"counts" if chart_num==1 else "pct"}.png'
        plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches='tight')
        print(f'Saved → {fname}')
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Per-class F1
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1(results: dict):
    modes = list(results.keys())
    x     = np.arange(len(GRADE_LABELS))
    width = 0.35 / max(len(modes) - 1, 1) * 2 if len(modes) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, mode in enumerate(modes):
        f1s    = f1_score(results[mode]['labels'], results[mode]['preds'],
                          average=None, zero_division=0)
        offset = (i - (len(modes) - 1) / 2) * (width + 0.04)
        bars   = ax.bar(x + offset, f1s, width,
                        label=MODEL_NAMES[mode],
                        color=MODEL_COLORS[mode],
                        edgecolor='white', linewidth=0.7, alpha=0.9)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_LABELS, fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score by Model', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label='0.9 target')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '03_per_class_f1.png', dpi=150, bbox_inches='tight')
    print('Saved → 03_per_class_f1.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Per-class Precision & Recall
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall(results: dict):
    modes = list(results.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    fig.suptitle('Per-Class Precision vs Recall', fontsize=14, fontweight='bold')

    x     = np.arange(len(GRADE_LABELS))
    width = 0.35

    for ax, mode in zip(axes, modes):
        r         = results[mode]
        precision = precision_score(r['labels'], r['preds'],
                                    average=None, zero_division=0)
        recall    = recall_score(r['labels'], r['preds'],
                                 average=None, zero_division=0)

        b1 = ax.bar(x - width/2, precision, width,
                    label='Precision', color='#3498db', edgecolor='white', alpha=0.85)
        b2 = ax.bar(x + width/2, recall,    width,
                    label='Recall',    color='#e74c3c', edgecolor='white', alpha=0.85)

        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(GRADE_LABELS, rotation=15, fontsize=10)
        ax.set_title(MODEL_NAMES[mode], fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '04_per_class_precision_recall.png',
                dpi=150, bbox_inches='tight')
    print('Saved → 04_per_class_precision_recall.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Overall metrics comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall_comparison(results: dict):
    modes   = list(results.keys())
    metrics = ['Accuracy', 'QW Kappa', 'Weighted F1', 'Weighted Precision', 'Weighted Recall']
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Overall Model Comparison', fontsize=14, fontweight='bold')

    for i, mode in enumerate(modes):
        r      = results[mode]
        vals   = [
            accuracy_score(r['labels'], r['preds']),
            r['kappa'],
            f1_score(r['labels'], r['preds'], average='weighted', zero_division=0),
            precision_score(r['labels'], r['preds'], average='weighted', zero_division=0),
            recall_score(r['labels'], r['preds'], average='weighted', zero_division=0),
        ]
        offset = (i - (len(modes)-1)/2) * (width + 0.02)
        bars   = ax.bar(x + offset, vals, width,
                        label=MODEL_NAMES[mode],
                        color=MODEL_COLORS[mode],
                        edgecolor='white', alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=0.9, color='gray', linestyle='--',
               linewidth=1, alpha=0.5)
    ax.text(len(metrics) - 0.5, 0.91, '0.90 target',
            color='gray', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '05_kappa_accuracy_comparison.png',
                dpi=150, bbox_inches='tight')
    print('Saved → 05_kappa_accuracy_comparison.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 — ROC curves (one-vs-rest per grade)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict, val_loader, val_dataset):
    """
    One-vs-Rest ROC curve per DR grade.
    Requires re-running inference to get softmax probabilities.
    """
    from torch.nn.functional import softmax as torch_softmax

    modes = list(results.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]
    fig.suptitle('ROC Curves — One-vs-Rest per Grade',
                 fontsize=14, fontweight='bold')

    for ax, mode in zip(axes, modes):
        model = load_model(mode)
        model.eval()
        val_dataset.use_captions = False

        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, text_features, labels, _ in val_loader:
                images        = images.to(device)
                text_features = text_features.to(device)
                dr_out, _     = model(images, text_features)
                probs         = torch_softmax(dr_out, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        bin_labels = label_binarize(all_labels, classes=list(range(NUM_DR_CLASSES)))

        for i, (name, color) in enumerate(zip(GRADE_LABELS, GRADE_COLORS)):
            fpr, tpr, _ = roc_curve(bin_labels[:, i], all_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name} (AUC={roc_auc:.2f})')

        ax.plot([0,1], [0,1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(MODEL_NAMES[mode], fontsize=12, fontweight='bold')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '06_roc_curves.png', dpi=150, bbox_inches='tight')
    print('Saved → 06_roc_curves.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7 — Prediction distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_distribution(results: dict):
    modes = list(results.keys())
    fig, axes = plt.subplots(1, len(modes) + 1,
                             figsize=(6 * (len(modes) + 1), 5))
    fig.suptitle('Prediction Distribution vs Ground Truth',
                 fontsize=14, fontweight='bold')

    # Ground truth in first panel
    gt_counts = np.bincount(results[modes[0]]['labels'], minlength=NUM_DR_CLASSES)
    axes[0].bar(GRADE_LABELS, gt_counts, color=GRADE_COLORS,
                edgecolor='white', alpha=0.9)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(gt_counts):
        axes[0].text(i, v + 3, str(v), ha='center', fontsize=10, fontweight='bold')

    for ax, mode in zip(axes[1:], modes):
        pred_counts = np.bincount(results[mode]['preds'],
                                  minlength=NUM_DR_CLASSES)
        ax.bar(GRADE_LABELS, pred_counts, color=GRADE_COLORS,
               edgecolor='white', alpha=0.9)
        ax.set_title(f'{MODEL_NAMES[mode]}\nPredictions',
                     fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        for i, v in enumerate(pred_counts):
            ax.text(i, v + 3, str(v), ha='center', fontsize=10, fontweight='bold')

    for ax in axes:
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '07_prediction_distribution.png',
                dpi=150, bbox_inches='tight')
    print('Saved → 07_prediction_distribution.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 8 — Error analysis heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_analysis(results: dict):
    """
    Shows which grades get confused with which other grades.
    Off-diagonal cells = errors. Diagonal = correct predictions.
    """
    modes = list(results.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]
    fig.suptitle('Error Analysis — Off-diagonal = Misclassifications',
                 fontsize=14, fontweight='bold')

    for ax, mode in zip(axes, modes):
        r  = results[mode]
        cm = confusion_matrix(r['labels'], r['preds'])

        # Zero out diagonal to highlight only errors
        error_matrix = cm.copy().astype(float)
        np.fill_diagonal(error_matrix, 0)

        # Normalise by true class total
        row_sums = cm.sum(axis=1, keepdims=True)
        error_pct = np.where(row_sums > 0,
                             error_matrix / row_sums * 100, 0)

        sns.heatmap(
            error_pct,
            annot=True, fmt='.1f', cmap='Reds',
            xticklabels=GRADE_LABELS,
            yticklabels=GRADE_LABELS,
            linewidths=0.5, ax=ax,
            vmin=0,
            annot_kws={'size': 10},
        )
        ax.set_title(
            f'{MODEL_NAMES[mode]}\n'
            f'(% of true class misclassified as each other grade)',
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel('Predicted as', fontsize=11)
        ax.set_ylabel('True Grade',   fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '08_error_analysis.png',
                dpi=150, bbox_inches='tight')
    print('Saved → 08_error_analysis.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save text reports
# ─────────────────────────────────────────────────────────────────────────────

def save_text_report(results: dict):
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    for mode in results:
        print_individual_report(mode, results)

    if len(results) > 1:
        print_side_by_side(results)

    sys.stdout = old_stdout
    report_text = buf.getvalue()

    path = RESULTS_DIR / 'evaluation_report.txt'
    path.write_text(report_text)
    print(f'Text report saved → {path}')
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(modes: list = None):
    """
    Runs full evaluation pipeline for all specified models.

    Args:
        modes : list of model names, e.g. ['convnext', 'image_only']
                default = both models
    """
    if modes is None:
        modes = ['image_only', 'convnext']

    print('=' * 58)
    print('  DR MODEL EVALUATION')
    print('=' * 58)
    print(f'  Models   : {[MODEL_NAMES[m] for m in modes]}')
    print(f'  Results  : {RESULTS_DIR}')
    print('=' * 58)

    # ── Data ──────────────────────────────────────────────────────────────────
    print('\nLoading validation data...')
    _, val_dataset, _, val_loader = prepare_data()

    # ── Inference ─────────────────────────────────────────────────────────────
    print('\nRunning inference...')
    results = get_all_results(modes, val_loader, val_dataset)

    # ── Text reports ──────────────────────────────────────────────────────────
    print('\n' + '─' * 58)
    print('  TEXT REPORTS')
    print('─' * 58)
    for mode in modes:
        print_individual_report(mode, results)

    if len(modes) > 1:
        print_side_by_side(results)

    save_text_report(results)

    # ── Charts ────────────────────────────────────────────────────────────────
    print('\n' + '─' * 58)
    print('  GENERATING CHARTS')
    print('─' * 58)

    plot_confusion_matrices(results)
    plot_per_class_f1(results)
    plot_precision_recall(results)
    plot_overall_comparison(results)
    plot_roc_curves(results, val_loader, val_dataset)
    plot_prediction_distribution(results)
    plot_error_analysis(results)

    # ── Final summary ─────────────────────────────────────────────────────────
    print('\n' + '═' * 58)
    print('  DONE')
    print('═' * 58)
    for mode in modes:
        acc = accuracy_score(results[mode]['labels'],
                             results[mode]['preds']) * 100
        print(f'  {MODEL_NAMES[mode]:<20} '
              f'Kappa: {results[mode]["kappa"]:.4f}  |  '
              f'Acc: {acc:.2f}%')
    print(f'\n  8 charts saved to: {RESULTS_DIR}')
    print('═' * 58)

    return results


if __name__ == '__main__':
    evaluate_all()
