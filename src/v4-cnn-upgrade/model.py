"""
model.py
────────
InceptionV3-based classifier for Diabetic Retinopathy grading (0–4).

Architecture (paper's exact spec):
  InceptionV3 backbone (ImageNet pretrained, 48 layers)
        ↓
  GlobalAveragePooling2D
        ↓
  Dropout
        ↓
  Dense(2048, ReLU)
        ↓
  Dropout
        ↓
  Dense(5, Softmax)    ← grades 0–4
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from config import INPUT_SHAPE, NUM_CLASSES, WARMUP_LR, FINETUNE_LR


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(freeze_backbone: bool = True) -> Model:
    """
    Builds the InceptionV3 DR classifier.

    Args:
        freeze_backbone : True for Phase 1 (warmup), False for Phase 2 (fine-tune)

    Returns:
        Compiled Keras model
    """

    # ── Base model: InceptionV3 pretrained on ImageNet ────────────────────────
    base_model = InceptionV3(
        weights     = 'imagenet',
        include_top = False,           # remove ImageNet classifier head
        input_shape = INPUT_SHAPE,
    )
    base_model.trainable = not freeze_backbone

    if freeze_backbone:
        print(f'InceptionV3 backbone FROZEN ✅  — Phase 1 (warmup)')
    else:
        print(f'InceptionV3 backbone UNFROZEN ✅ — Phase 2 (fine-tuning)')

    # ── Custom classification head (paper's architecture) ─────────────────────
    x = base_model.output
    x = GlobalAveragePooling2D()(x)          # reduces spatial dims, more robust
    x = Dropout(0.5)(x)                      # prevent overfitting
    x = Dense(2048, activation='relu')(x)    # paper: 2048 units with ReLU
    x = Dropout(0.5)(x)                      # second dropout before output
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # 5-class probability

    model = Model(inputs=base_model.input, outputs=outputs)

    return model


def compile_model(model: Model, phase: int = 1) -> Model:
    """
    Compiles the model with the correct LR for each phase.

    Phase 1 (warmup)    : lr = 1e-3  (higher — only head trains)
    Phase 2 (fine-tune) : lr = 1e-5  (very low — full model trains)

    Args:
        model : Keras Model
        phase : 1 or 2

    Returns:
        Compiled model
    """
    lr = WARMUP_LR if phase == 1 else FINETUNE_LR

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )

    print(f'Model compiled — Phase {phase} | LR: {lr:.0e}')
    return model


def get_model_summary(model: Model) -> None:
    """Prints parameter counts split by trainable/frozen."""
    trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    frozen = sum(
        tf.size(w).numpy() for w in model.non_trainable_weights
    )
    print(f'  Trainable params : {trainable:,}')
    print(f'  Frozen params    : {frozen:,}')
    print(f'  Total params     : {trainable + frozen:,}')


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    print('── Phase 1 model ──')
    model = build_model(freeze_backbone=True)
    model = compile_model(model, phase=1)
    get_model_summary(model)

    # Forward pass test
    dummy = np.random.rand(2, 299, 299, 3).astype(np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f'\nOutput shape : {out.shape}')   # (2, 5)
    print(f'Sum of probs : {out.sum(axis=1)}')  # should be ~[1.0, 1.0]
    print('\nmodel.py ✅')
