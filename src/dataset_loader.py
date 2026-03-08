"""
dataset_loader.py
─────────────────
PyTorch Dataset and DataLoader factory for the APTOS 2019 dataset.

Handles:
  - Loading preprocessed images from data/processed/
  - Train / Validation / Test splitting (70 / 15 / 15)
  - Data augmentation for training set
  - Returns ready-to-use DataLoaders for model training

Usage:
  from src.dataset_loader import get_dataloaders

  train_loader, val_loader, test_loader = get_dataloaders()
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / 'data' / 'processed'

# ── Normalisation constants (ImageNet — used by pretrained EfficientNet) ──────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 retinal images.

    Args:
        df        : DataFrame with columns [id_code, diagnosis]
        img_dir   : Path to folder containing preprocessed .png images
        transform : torchvision transforms to apply
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
        image = Image.open(str(path)).convert('RGB')
        label = int(row['diagnosis'])

        if self.transform:
            image = self.transform(image)

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int = 224) -> dict:
    """
    Returns a dict of transforms for each split.

    Training  : augmentation (flips, rotation, colour jitter, blur)
    Validation: no augmentation, only resize + normalise
    Test      : same as validation
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        'train': train_tf,
        'val'  : eval_tf,
        'test' : eval_tf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    proc_dir   : Path = PROC_DIR,
    img_size   : int  = 224,
    batch_size : int  = 32,
    num_workers: int  = 4,
    val_size   : float = 0.15,
    test_size  : float = 0.15,
    random_seed: int  = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the dataset and returns (train_loader, val_loader, test_loader).

    Split: 70% train / 15% val / 15% test  (stratified by diagnosis grade)

    Args:
        proc_dir    : Path to data/processed/
        img_size    : Image size expected by the model (must match preprocessing)
        batch_size  : Number of images per batch
        num_workers : Parallel workers for data loading (set 0 on Windows if issues)
        val_size    : Fraction of data for validation
        test_size   : Fraction of data for test
        random_seed : Reproducibility seed

    Returns:
        (train_loader, val_loader, test_loader)
    """

    csv_path = proc_dir / 'train.csv'
    img_dir  = proc_dir / 'train_images'

    if not csv_path.exists():
        raise FileNotFoundError(f'train.csv not found at {csv_path}')
    if not img_dir.exists():
        raise FileNotFoundError(f'Processed images not found at {img_dir}')

    df = pd.read_csv(csv_path)

    # ── Stratified split: train / temp (val + test) ───────────────────────────
    temp_frac = val_size + test_size
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_frac,
        stratify=df['diagnosis'],
        random_state=random_seed,
    )

    # Split temp into val and test (equal halves if val_size == test_size)
    val_frac_of_temp = val_size / temp_frac
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - val_frac_of_temp,
        stratify=df_temp['diagnosis'],
        random_state=random_seed,
    )

    print(f'Split summary:')
    print(f'  Train : {len(df_train)} images ({len(df_train)/len(df)*100:.1f}%)')
    print(f'  Val   : {len(df_val)}   images ({len(df_val)/len(df)*100:.1f}%)')
    print(f'  Test  : {len(df_test)}  images ({len(df_test)/len(df)*100:.1f}%)')
    print()

    # ── Transforms ────────────────────────────────────────────────────────────
    tfs = get_transforms(img_size)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = APTOSDataset(df_train, img_dir, transform=tfs['train'])
    val_ds   = APTOSDataset(df_val,   img_dir, transform=tfs['val'])
    test_ds  = APTOSDataset(df_test,  img_dir, transform=tfs['test'])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # pin_memory speeds up CPU→GPU transfers
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

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — run directly to verify everything works
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Testing dataset_loader...\n')

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_workers=0,   # safer for a quick test on Windows
    )

    # Grab one batch and print shapes
    images, labels = next(iter(train_loader))
    print(f'Batch image shape : {images.shape}')   # (8, 3, 224, 224)
    print(f'Batch labels      : {labels}')
    print(f'Label dtype       : {labels.dtype}')
    print(f'\nTrain batches : {len(train_loader)}')
    print(f'Val   batches : {len(val_loader)}')
    print(f'Test  batches : {len(test_loader)}')
    print('\ndataset_loader.py ✅')