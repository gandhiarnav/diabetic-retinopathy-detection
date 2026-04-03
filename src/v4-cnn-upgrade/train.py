"""
train.py
────────
Main training script — follows the paper's exact two-phase strategy.

Phase 1 — Warmup (5 epochs):
  Backbone frozen, only classification head trains.
  High LR (1e-3). Fast — establishes good head weights.

Phase 2 — Fine-tuning (30 epochs):
  Full model unfrozen, very low LR (1e-5).
  ModelCheckpoint + EarlyStopping + ReduceLROnPlateau active.

Usage (Kaggle notebook):
  Run cells sequentially, or:
  !python train.py

Saves:
  /kaggle/working/models/best_model.keras
  /kaggle/working/results/training_curves.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from config import (
    WARMUP_EPOCHS, FINETUNE_EPOCHS,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, MIN_LR,
    MODEL_DIR, RESULTS_DIR,
)
from data_loader import prepare_data
from model import build_model, compile_model, get_model_summary


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(phase: int) -> list:
    """
    Returns Keras callbacks for each training phase.

    Phase 1: Only ModelCheckpoint (short warmup, no early stopping needed)
    Phase 2: Full set — ModelCheckpoint + EarlyStopping + ReduceLROnPlateau
    """
    ckpt_path = str(MODEL_DIR / 'best_model.keras')

    checkpoint = ModelCheckpoint(
        filepath          = ckpt_path,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        save_weights_only = False,
        mode              = 'max',
        verbose           = 1,
    )

    if phase == 1:
        return [checkpoint]

    # Phase 2 — full callback set
    early_stop = EarlyStopping(
        monitor              = 'val_loss',
        patience             = EARLY_STOP_PATIENCE,
        restore_best_weights = True,   # revert to best weights on stop
        verbose              = 1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = REDUCE_LR_FACTOR,   # multiply LR by 0.3
        patience = REDUCE_LR_PATIENCE,
        min_lr   = MIN_LR,
        verbose  = 1,
    )

    return [checkpoint, early_stop, reduce_lr]


# ─────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history_p1, history_p2) -> None:
    """
    Plots loss and accuracy curves for both phases combined.
    Saves to results/training_curves.png
    """
    # Concatenate phase histories
    train_loss = history_p1['loss']      + history_p2['loss']
    val_loss   = history_p1['val_loss']  + history_p2['val_loss']
    train_acc  = history_p1['accuracy']  + history_p2['accuracy']
    val_acc    = history_p1['val_accuracy'] + history_p2['val_accuracy']

    p1_end = len(history_p1['loss'])   # epoch where phase 2 begins
    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — InceptionV3 DR Classifier',
                 fontsize=14, fontweight='bold')

    colors = {'train': '#2196F3', 'val': '#F44336'}

    for ax, train_vals, val_vals, ylabel, title in zip(
        axes,
        [train_loss, train_acc],
        [val_loss,   val_acc],
        ['Loss', 'Accuracy'],
        ['Cross-Entropy Loss', 'Accuracy'],
    ):
        ax.plot(epochs, train_vals, color=colors['train'],
                label='Train', linewidth=2)
        ax.plot(epochs, val_vals,   color=colors['val'],
                label='Val',   linewidth=2)
        ax.axvline(x=p1_end + 0.5, color='gray', linestyle='--',
                   linewidth=1.5, label='Phase 1 → 2')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nCurves saved → {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:

    # ── GPU check ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPUs available : {len(gpus)}')
    for g in gpus:
        print(f'  {g}')
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print('Preparing data...')
    train_gen, val_gen, test_gen, df_test = prepare_data()
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Warmup
    # Frozen backbone, train head only, 5 epochs
    # ═════════════════════════════════════════════════════════════════════════
    print('─' * 55)
    print('  PHASE 1 — Warmup (frozen backbone)')
    print('─' * 55)

    model = build_model(freeze_backbone=True)
    model = compile_model(model, phase=1)
    get_model_summary(model)
    print()

    history_p1 = model.fit(
        train_gen,
        epochs          = WARMUP_EPOCHS,
        validation_data = val_gen,
        callbacks       = get_callbacks(phase=1),
        verbose         = 1,
    ).history

    print(f'\n  Phase 1 complete')
    print(f'  Best val accuracy : {max(history_p1["val_accuracy"]):.4f}')

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Fine-tuning
    # Unfreeze entire model, very low LR, 30 epochs
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print('─' * 55)
    print('  PHASE 2 — Fine-tuning (unfrozen backbone)')
    print('─' * 55)

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    # Recompile with low LR
    model = compile_model(model, phase=2)
    get_model_summary(model)
    print()

    history_p2 = model.fit(
        train_gen,
        epochs          = FINETUNE_EPOCHS,
        validation_data = val_gen,
        callbacks       = get_callbacks(phase=2),
        verbose         = 1,
    ).history

    print(f'\n  Phase 2 complete')
    print(f'  Best val accuracy : {max(history_p2["val_accuracy"]):.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_curves(history_p1, history_p2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print('═' * 55)
    print('  TRAINING COMPLETE')
    print('═' * 55)
    print(f'  Phase 1 best val acc : {max(history_p1["val_accuracy"])*100:.2f}%')
    print(f'  Phase 2 best val acc : {max(history_p2["val_accuracy"])*100:.2f}%')
    print(f'  Model saved          : {MODEL_DIR}/best_model.keras')
    print('═' * 55)


if __name__ == '__main__':
    train()
