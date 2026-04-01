"""
dataset_loader.py
─────────────────
PyTorch Dataset and DataLoader factory for the combined APTOS + IDRiD dataset.

Key improvements over v1:
  - Oversampling  : minority classes upsampled to match majority class
  - 360° rotation : retinal images have no natural orientation
  - GridDistortion: simulates eye lens curvature variation
  - RandomGamma   : handles different fundus camera exposures
  - Loads unified train.csv (APTOS + IDRiD combined)

Usage:
  from src.dataset_loader import get_dataloaders
  train_loader, val_loader, test_loader = get_dataloaders()
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / 'data' / 'processed'

# ── ImageNet normalisation (used by pretrained EfficientNet) ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RetinalDataset(Dataset):
    """
    PyTorch Dataset for combined APTOS + IDRiD retinal images.

    Args:
        df        : DataFrame with columns [id_code, diagnosis]
        img_dir   : Path to folder containing preprocessed .png images
        transform : albumentations transform pipeline
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        path  = self.img_dir / f"{row['id_code']}.png"
        image = np.array(Image.open(str(path)).convert('RGB'))
        label = int(row['diagnosis'])

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Oversampling
# ─────────────────────────────────────────────────────────────────────────────

def oversample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversamples minority classes so every grade has equal representation.
    Matches the paper's approach: all classes upsampled to majority class count.

    Only applied to training set — val and test are never oversampled.

    Example:
      Before: No DR=1263, Mild=182, Moderate=549, Severe=96, Prolif=145
      After : All classes = 1263  →  total = 6315
    """
    max_count = df['diagnosis'].value_counts().max()
    balanced  = []

    for grade in range(5):
        grade_df = df[df['diagnosis'] == grade]
        if len(grade_df) == 0:
            continue
        # Sample with replacement to reach max_count
        oversampled = grade_df.sample(
            n=max_count, replace=True, random_state=42
        )
        balanced.append(oversampled)

    balanced_df = pd.concat(balanced).sample(frac=1, random_state=42)  # shuffle
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (albumentations)
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int = 224) -> dict:
    """
    Returns albumentations transform pipelines for each split.

    Training improvements over v1:
      - 360° rotation (retinas have no orientation)
      - GridDistortion (simulates eye lens curvature)
      - RandomGamma (handles different camera exposures)
      - CoarseDropout (simulates occluded regions)

    Val/Test: only resize + normalise (no augmentation)
    """

    train_tf = A.Compose([
        A.Resize(img_size, img_size),

        # ── Geometric ─────────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=360, p=0.8),              # full 360° — paper's approach
        A.GridDistortion(                         # eye lens curvature simulation
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),

        # ── Colour / lighting ─────────────────────────────────────────────────
        A.RandomGamma(                            # camera exposure variation
            gamma_limit=(80, 120),
            p=0.4
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            p=0.4
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Regularisation ────────────────────────────────────────────────────
        A.CoarseDropout(                          # randomly masks small patches
            max_holes=8,
            max_height=16,
            max_width=16,
            p=0.2
        ),

        # ── Normalise & tensor ────────────────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    eval_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return {'train': train_tf, 'val': eval_tf, 'test': eval_tf}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    proc_dir   : Path  = PROC_DIR,
    img_size   : int   = 224,
    batch_size : int   = 32,
    num_workers: int   = 0,
    val_size   : float = 0.15,
    test_size  : float = 0.15,
    random_seed: int   = 42,
    oversample : bool  = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads unified APTOS + IDRiD CSV, splits, oversamples train set,
    and returns (train_loader, val_loader, test_loader).

    Split is stratified by diagnosis grade.
    Oversampling is applied ONLY to train set.
    """

    csv_path = proc_dir / 'train.csv'
    img_dir  = proc_dir / 'train_images'

    if not csv_path.exists():
        raise FileNotFoundError(
            f'Unified train.csv not found at {csv_path}\n'
            f'Run data_preprocessing.py first.'
        )

    df = pd.read_csv(csv_path)

    # ── Stratified split ──────────────────────────────────────────────────────
    temp_frac = val_size + test_size
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_frac,
        stratify=df['diagnosis'],
        random_state=random_seed,
    )
    val_frac_of_temp = val_size / temp_frac
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - val_frac_of_temp,
        stratify=df_temp['diagnosis'],
        random_state=random_seed,
    )

    print(f'Split (before oversampling):')
    print(f'  Train : {len(df_train)}')
    print(f'  Val   : {len(df_val)}')
    print(f'  Test  : {len(df_test)}')

    # ── Oversample train set ──────────────────────────────────────────────────
    if oversample:
        df_train_balanced = oversample_dataframe(df_train)
        print(f'\nAfter oversampling:')
        print(f'  Train : {len(df_train_balanced)}')
        for g in range(5):
            c = len(df_train_balanced[df_train_balanced['diagnosis'] == g])
            print(f'    Grade {g}: {c}')
    else:
        df_train_balanced = df_train

    # ── Transforms ────────────────────────────────────────────────────────────
    tfs = get_transforms(img_size)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = RetinalDataset(df_train_balanced, img_dir, transform=tfs['train'])
    val_ds   = RetinalDataset(df_val,            img_dir, transform=tfs['val'])
    test_ds  = RetinalDataset(df_test,           img_dir, transform=tfs['test'])

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(f'\nTrain batches : {len(train_loader)}')
    print(f'Val   batches : {len(val_loader)}')
    print(f'Test  batches : {len(test_loader)}')

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Testing dataset_loader...\n')

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))
    print(f'\nBatch image shape : {images.shape}')
    print(f'Batch labels      : {labels}')
    print('\ndataset_loader.py ✅')
