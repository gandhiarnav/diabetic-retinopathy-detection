"""
data_loader.py
──────────────
Prepares balanced datasets and Keras ImageDataGenerators
for training the InceptionV3 DR classifier.

Key functions:
  balance_dataset()  : oversample minority classes to match majority
  get_generators()   : Keras ImageDataGenerator with paper's augmentations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    IMG_DIR, CSV_PATH, IMG_SIZE, BATCH_SIZE,
    NUM_CLASSES, CLASS_NAMES, RANDOM_SEED,
    VAL_SPLIT, TEST_SPLIT, OVERSAMPLE_TARGET,
    ROTATION_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Oversampling
# ─────────────────────────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample minority DR classes so every grade has equal representation.

    Matches the paper exactly:
      All classes upsampled to majority class count (No DR = 1805)
      → total balanced training set ≈ 9,025 images

    IMPORTANT: Only call this on the training split.
               Val and test sets must never be oversampled.

    Args:
        df : DataFrame with columns [filename, diagnosis]

    Returns:
        Balanced DataFrame with equal class counts
    """
    target = OVERSAMPLE_TARGET or df['diagnosis'].value_counts().max()
    balanced_parts = []

    for grade in range(NUM_CLASSES):
        grade_df = df[df['diagnosis'] == grade]
        if len(grade_df) == 0:
            continue
        oversampled = grade_df.sample(
            n=target, replace=True, random_state=RANDOM_SEED
        )
        balanced_parts.append(oversampled)

    balanced = (pd.concat(balanced_parts)
                  .sample(frac=1, random_state=RANDOM_SEED)  # shuffle
                  .reset_index(drop=True))

    print(f'  Balanced train set: {len(balanced)} images')
    print(f'  Per class         : {target} each')
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads unified CSV, splits into train/val/test (stratified).
    Returns (df_train_raw, df_val, df_test) — train is NOT yet oversampled.

    Split:
      Train : 70%  (will be oversampled to balance classes)
      Val   : 15%  (no oversampling — reflects real distribution)
      Test  : 15%  (no oversampling — reflects real distribution)
    """
    df = pd.read_csv(CSV_PATH)

    # Build filename column (id_code + .png)
    df['filename']  = df['id_code'] + '.png'
    df['diagnosis'] = df['diagnosis'].astype(str)  # Keras flow needs string labels

    # Stratified split
    temp_size = VAL_SPLIT + TEST_SPLIT
    df_train, df_temp = train_test_split(
        df, test_size=temp_size,
        stratify=df['diagnosis'],
        random_state=RANDOM_SEED,
    )
    val_frac = VAL_SPLIT / temp_size
    df_val, df_test = train_test_split(
        df_temp, test_size=1 - val_frac,
        stratify=df_temp['diagnosis'],
        random_state=RANDOM_SEED,
    )

    # Convert diagnosis back to int for oversampling
    df_train = df_train.copy()
    df_train['diagnosis'] = df_train['diagnosis'].astype(int)
    df_val   = df_val.copy()
    df_val['diagnosis']   = df_val['diagnosis'].astype(int)
    df_test  = df_test.copy()
    df_test['diagnosis']  = df_test['diagnosis'].astype(int)

    print(f'Split:')
    print(f'  Train (raw) : {len(df_train)}')
    print(f'  Val         : {len(df_val)}')
    print(f'  Test        : {len(df_test)}')
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# Keras ImageDataGenerators
# ─────────────────────────────────────────────────────────────────────────────

def get_generators(
    df_train : pd.DataFrame,
    df_val   : pd.DataFrame,
    df_test  : pd.DataFrame,
    img_dir  : Path = IMG_DIR,
) -> tuple:
    """
    Creates Keras ImageDataGenerators for train, val, and test sets.

    Training augmentations (paper's exact spec):
      - Rescale to [0, 1]
      - Random rotation up to 360°
      - Random horizontal flip
      - Random vertical flip

    Val/Test:
      - Rescale only — no augmentation

    Args:
        df_train : balanced training DataFrame
        df_val   : validation DataFrame
        df_test  : test DataFrame
        img_dir  : path to folder with preprocessed images

    Returns:
        (train_gen, val_gen, test_gen)
    """

    # ── Convert diagnosis to string for Keras flow_from_dataframe ─────────────
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()

    for d in [df_train, df_val, df_test]:
        d['diagnosis'] = d['diagnosis'].astype(str)

    # ── Training generator — with augmentation ────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale            = 1.0 / 255.0,
        rotation_range     = ROTATION_RANGE,
        horizontal_flip    = HORIZONTAL_FLIP,
        vertical_flip      = VERTICAL_FLIP,
        fill_mode          = 'reflect',        # fills gaps after rotation naturally
    )

    # ── Val / Test generator — rescale only ───────────────────────────────────
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # ── Flow from DataFrame ───────────────────────────────────────────────────
    common_kwargs = dict(
        directory   = str(img_dir),
        x_col       = 'filename',
        y_col       = 'diagnosis',
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        color_mode  = 'rgb',
    )

    train_gen = train_datagen.flow_from_dataframe(
        df_train, shuffle=True, seed=RANDOM_SEED, **common_kwargs
    )
    val_gen = eval_datagen.flow_from_dataframe(
        df_val, shuffle=False, **common_kwargs
    )
    test_gen = eval_datagen.flow_from_dataframe(
        df_test, shuffle=False, **common_kwargs
    )

    print(f'\nGenerators ready:')
    print(f'  Train batches : {len(train_gen)}')
    print(f'  Val   batches : {len(val_gen)}')
    print(f'  Test  batches : {len(test_gen)}')

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — combine all steps
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data() -> tuple:
    """
    One-call function: load → split → balance → generators.
    Returns (train_gen, val_gen, test_gen, df_test)
    df_test is returned for use in evaluate.py
    """
    df_train_raw, df_val, df_test = load_and_split()

    print('\nBalancing training set...')
    df_train_balanced = balance_dataset(df_train_raw)

    print('\nCreating generators...')
    train_gen, val_gen, test_gen = get_generators(
        df_train_balanced, df_val, df_test
    )

    return train_gen, val_gen, test_gen, df_test
