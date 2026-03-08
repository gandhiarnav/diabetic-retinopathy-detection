"""
data_preprocessing.py
─────────────────────
Reads raw APTOS 2019 retinal images and saves cleaned, resized versions
to data/processed/ ready for model training.

Pipeline per image:
  1. Load image
  2. Crop black borders
  3. Resize to target size (default 224×224)
  4. Apply Ben Graham normalisation (removes lighting artefacts)
  5. Save to data/processed/train_images/

Usage:
  python src/data_preprocessing.py
  python src/data_preprocessing.py --size 512 --input data/raw --output data/processed
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
RAW_DIR    = BASE_DIR / 'data' / 'raw'
PROC_DIR   = BASE_DIR / 'data' / 'processed'
IMG_SIZE   = 224   # pixels (square)


# ─────────────────────────────────────────────────────────────────────────────
# Core image functions
# ─────────────────────────────────────────────────────────────────────────────

def crop_black_borders(img: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Remove the dark circular border common in fundus photographs.
    Crops rows/cols where all pixel values are below `tolerance`.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mask = gray > tolerance                    # True where there is content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():       # all-black image guard
        return img

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return img[r_min:r_max + 1, c_min:c_max + 1]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image to (size × size)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ben_graham_normalise(img: np.ndarray, size: int) -> np.ndarray:
    """
    Ben Graham's normalisation technique:
      output = 4 * img - 4 * GaussianBlur(img) + 128
    This removes low-frequency lighting variations and enhances
    fine vascular structures that indicate DR.
    """
    blur_radius = max(1, size // 30)           # scale sigma with image size
    # kernel size must be odd
    k = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(img, (k, k), blur_radius)
    normalised = cv2.addWeighted(img, 4, blurred, -4, 128)
    return normalised


def preprocess_image(src_path: Path, size: int = IMG_SIZE) -> np.ndarray | None:
    """
    Full preprocessing pipeline for a single image.
    Returns a uint8 numpy array of shape (size, size, 3) or None on failure.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return None

    img = crop_black_borders(img)
    img = resize(img, size)
    img = ben_graham_normalise(img, size)

    # Clip to valid uint8 range (Ben Graham can push values outside 0-255)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    raw_dir:  Path = RAW_DIR,
    proc_dir: Path = PROC_DIR,
    size:     int  = IMG_SIZE,
) -> None:

    raw_img_dir  = raw_dir  / 'train_images'
    proc_img_dir = proc_dir / 'train_images'
    csv_path     = raw_dir  / 'train.csv'

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not raw_img_dir.exists():
        raise FileNotFoundError(f'Raw image folder not found: {raw_img_dir}')
    if not csv_path.exists():
        raise FileNotFoundError(f'train.csv not found: {csv_path}')

    # ── Output folder ─────────────────────────────────────────────────────────
    proc_img_dir.mkdir(parents=True, exist_ok=True)

    # ── Load labels ───────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f'Images to process : {len(df)}')
    print(f'Output folder     : {proc_img_dir}')
    print(f'Target size       : {size}×{size}px')
    print()

    # ── Process ───────────────────────────────────────────────────────────────
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Preprocessing'):
        src  = raw_img_dir  / f"{row['id_code']}.png"
        dest = proc_img_dir / f"{row['id_code']}.png"

        # Skip if already processed (useful for resuming interrupted runs)
        if dest.exists():
            continue

        result = preprocess_image(src, size=size)

        if result is None:
            failed.append(row['id_code'])
            continue

        cv2.imwrite(str(dest), result)

    # ── Report ────────────────────────────────────────────────────────────────
    processed = len(list(proc_img_dir.glob('*.png')))
    print()
    print('=' * 45)
    print('  PREPROCESSING COMPLETE')
    print('=' * 45)
    print(f'  Processed : {processed}')
    print(f'  Failed    : {len(failed)}')
    print(f'  Skipped   : {len(df) - processed - len(failed)} (already existed)')

    if failed:
        print(f'\n  ⚠️  Failed images:')
        for fid in failed:
            print(f'    {fid}')

        # Save failed list for inspection
        fail_path = proc_dir / 'failed_images.txt'
        fail_path.write_text('\n'.join(failed))
        print(f'\n  Saved failed list → {fail_path}')

    # Copy train.csv to processed/ so the dataset loader can find labels
    dest_csv = proc_dir / 'train.csv'
    if not dest_csv.exists():
        df.to_csv(dest_csv, index=False)
        print(f'\n  Copied train.csv → {dest_csv}')

    print('=' * 45)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Preprocess APTOS 2019 retinal images')
    p.add_argument('--size',   type=int,  default=IMG_SIZE,
                   help=f'Output image size in pixels (default: {IMG_SIZE})')
    p.add_argument('--input',  type=Path, default=RAW_DIR,
                   help='Path to data/raw directory')
    p.add_argument('--output', type=Path, default=PROC_DIR,
                   help='Path to data/processed directory')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_dataset(
        raw_dir  = args.input,
        proc_dir = args.output,
        size     = args.size,
    )