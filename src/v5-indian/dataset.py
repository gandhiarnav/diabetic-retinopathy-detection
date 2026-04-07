"""
dataset.py
──────────
Dataset class and data preparation for the IDRiD DR classifier.

Key insight from the notebook:
  The overlay images (IDRiD_001_L14.jpg, IDRiD_001_L20.jpg etc.)
  are NOT discarded — they are used as extra training samples with
  the same label as their base image. This turns 413 images → 3,304.

  Val set uses ORIGINAL images only (no overlays).
"""

import os
import glob
import torch
import torchvision
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import CLIPTokenizer, CLIPTextModel

from config import (
    IMAGE_DIR, CSV_PATH,
    IMG_SIZE, BATCH_SIZE,
    RANDOM_SEED, VAL_SPLIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP text encoder (frozen) — used only when use_captions=True
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_text_model():
    """
    Loads frozen CLIP text encoder for caption embeddings.
    Kept on CPU to save GPU VRAM.
    """
    tokenizer   = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    text_model  = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').cpu()
    for param in text_model.parameters():
        param.requires_grad = False
    return tokenizer, text_model


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DRDataset(Dataset):
    """
    Dataset for IDRiD retinal images.

    Each item returns:
      image        : (3, 224, 224) float tensor
      text_features: (512,) CLIP embedding or zeros if captions disabled
      label        : DR grade (0–4)
      edema_label  : macular edema risk (0–2)

    use_captions can be toggled externally:
      dataset.use_captions = True   # enables CLIP text features
      dataset.use_captions = False  # zeros for text (image-only mode)
    """

    def __init__(
        self,
        dataframe  : pd.DataFrame,
        image_dir  : Path,
        tokenizer  = None,
        text_model = None,
    ):
        self.df           = dataframe.reset_index(drop=True)
        self.image_dir    = str(image_dir)
        self.tokenizer    = tokenizer
        self.text_model   = text_model
        self.use_captions = False   # toggled externally
        self.resize       = torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, row['image'])
        image    = torchvision.io.read_image(img_path).float() / 255.0
        image    = self.resize(image)   # (3, 224, 224)

        # ── Text features ─────────────────────────────────────────────────────
        if self.use_captions and self.tokenizer is not None:
            tokens = self.tokenizer(
                row['caption'],
                return_tensors  = 'pt',
                padding         = 'max_length',
                max_length      = 77,
                truncation      = True,
            )
            with torch.no_grad():
                text_features = self.text_model(
                    input_ids      = tokens['input_ids'],
                    attention_mask = tokens['attention_mask'],
                ).pooler_output.squeeze(0)   # (512,)
        else:
            text_features = torch.zeros(512)

        label       = torch.tensor(int(row['level']),  dtype=torch.long)
        edema_label = torch.tensor(int(row['edema']),  dtype=torch.long)

        return image, text_features, label, edema_label


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_csv() -> pd.DataFrame:
    """Loads and standardises the IDRiD CSV."""
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={
        'Image name'           : 'image',
        'Retinopathy grade'    : 'level',
        'Risk of macular edema': 'edema',
        'class'                : 'merged_class',
        'Captions'             : 'caption',
    })
    df['image'] = df['image'] + '.jpg'
    return df


def get_base_id(filename: str) -> str:
    """
    Extracts base image ID from any filename variant.
    e.g.  IDRiD_001_L14.jpg  →  IDRiD_001
          IDRiD_001.jpg       →  IDRiD_001
    """
    return '_'.join(filename.replace('.jpg', '').split('_')[:2])


def expand_with_augments(split_df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """
    Expands a DataFrame to include overlay annotation images as extra samples.

    IDRiD provides lesion overlay variants per image:
      IDRiD_001.jpg      ← original
      IDRiD_001_L14.jpg  ← lesion overlay (same label)
      IDRiD_001_L20.jpg  ← lesion overlay (same label)

    All variants get the same DR grade as the original.
    This is how 413 original images become 3,304 training samples.

    Only called for TRAIN split — val uses originals only.
    """
    all_images     = glob.glob(os.path.join(str(image_dir), '*.jpg'))
    all_img_names  = [os.path.basename(p) for p in all_images]
    original_ids   = set(split_df['image'].str.replace('.jpg', '', regex=False))

    expanded_rows = []
    for img_name in all_img_names:
        base_id = get_base_id(img_name)
        if base_id in original_ids:
            match = split_df[split_df['image'].str.startswith(base_id)]
            if not match.empty:
                row          = match.iloc[0].copy()
                row['image'] = img_name
                expanded_rows.append(row)

    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def prepare_data(
    use_clip : bool = False,
) -> tuple[DRDataset, DRDataset, DataLoader, DataLoader]:
    """
    Full data pipeline:
      1. Load CSV
      2. Split into train (80%) / val (20%) — stratified
      3. Expand train with overlay augments
      4. Create Datasets and DataLoaders

    Args:
        use_clip : if True, load CLIP model for caption embeddings

    Returns:
        (train_dataset, val_dataset, train_loader, val_loader)
    """
    df = load_csv()

    # ── Stratified split on ORIGINAL images only ──────────────────────────────
    train_orig, val_orig = train_test_split(
        df,
        test_size  = VAL_SPLIT,
        stratify   = df['level'],
        random_state = RANDOM_SEED,
    )

    # ── Expand train with overlays — val stays original-only ──────────────────
    train_df = expand_with_augments(train_orig, IMAGE_DIR)
    val_df   = val_orig.reset_index(drop=True)

    print(f'Train (with overlays) : {len(train_df)}')
    print(f'Val   (originals only): {len(val_df)}')
    print(f'\nTrain class distribution:')
    print(train_df['level'].value_counts().sort_index().to_string())

    # ── CLIP (optional) ───────────────────────────────────────────────────────
    tokenizer, text_model = (load_clip_text_model() if use_clip else (None, None))

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = DRDataset(train_df, IMAGE_DIR, tokenizer, text_model)
    val_dataset   = DRDataset(val_df,   IMAGE_DIR, tokenizer, text_model)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f'\nTrain batches : {len(train_loader)}')
    print(f'Val   batches : {len(val_loader)}')

    return train_dataset, val_dataset, train_loader, val_loader
