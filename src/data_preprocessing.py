"""
data_preprocessing.py
─────────────────────
Preprocesses APTOS and IDRiD retinal images into a unified
processed dataset ready for training.

Pipeline per image:
  1. Load image
  2. Crop black borders
  3. Resize to target size (default 224×224)
  4. CLAHE on LAB luminance channel  ← medical imaging enhancement
  5. Ben Graham normalisation
  6. Save to data/processed/train_images/

Usage:
  python src/data_preprocessing.py
  python src/data_preprocessing.py --size 224
"""

import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / 'data' / 'raw'
PROC_DIR  = BASE_DIR / 'data' / 'processed'
IMG_SIZE  = 224


# ─────────────────────────────────────────────────────────────────────────────
# Core image functions
# ─────────────────────────────────────────────────────────────────────────────

def crop_black_borders(img: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """Remove dark circular border common in fundus photographs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray > tolerance
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return img[r_min:r_max + 1, c_min:c_max + 1]


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)

    Applied to the luminance (L) channel of the LAB colour space.
    Enhances local contrast in dark regions where DR lesions hide
    without distorting colour information.

    Why LAB instead of directly on BGR:
      LAB separates luminance from colour — CLAHE on L channel
      boosts contrast without shifting reds/greens that indicate
      haemorrhages and microaneurysms.

    clipLimit=2.0  : prevents noise amplification
    tileGridSize   : 8×8 local regions for adaptive enhancement
    """
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab       = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    l_clahe   = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ben_graham_normalise(img: np.ndarray, size: int) -> np.ndarray:
    """
    Ben Graham normalisation — removes low-frequency lighting variation,
    enhances fine vascular structures.
    output = 4 * img - 4 * GaussianBlur(img) + 128
    """
    blur_radius = max(1, size // 30)
    k           = blur_radius * 2 + 1
    blurred     = cv2.GaussianBlur(img, (k, k), blur_radius)
    normalised  = cv2.addWeighted(img, 4, blurred, -4, 128)
    return normalised


def preprocess_image(src_path: Path, size: int = IMG_SIZE) -> np.ndarray | None:
    """
    Full preprocessing pipeline for a single image.
    Handles both .png (APTOS) and .jpg (IDRiD).
    Returns uint8 numpy array (size, size, 3) or None on failure.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return None

    img = crop_black_borders(img)
    img = resize(img, size)
    img = apply_clahe(img)
    img = ben_graham_normalise(img, size)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dataset processors
# ─────────────────────────────────────────────────────────────────────────────

def process_aptos(proc_img_dir: Path, size: int) -> tuple[int, list]:
    """Process APTOS images from data/raw/aptos/"""
    raw_img_dir = RAW_DIR / 'aptos' / 'train_images'
    csv_path    = RAW_DIR / 'aptos' / 'train.csv'

    if not raw_img_dir.exists():
        raise FileNotFoundError(f'APTOS images not found: {raw_img_dir}')

    df     = pd.read_csv(csv_path)
    failed = []
    count  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='  APTOS'):
        src  = raw_img_dir  / f"{row['id_code']}.png"
        dest = proc_img_dir / f"aptos_{row['id_code']}.png"  # prefix avoids ID clashes

        if dest.exists():
            count += 1
            continue

        result = preprocess_image(src, size=size)
        if result is None:
            failed.append(row['id_code'])
            continue

        cv2.imwrite(str(dest), result)
        count += 1

    return count, failed


def process_idrid(proc_img_dir: Path, size: int) -> tuple[int, list]:
    """
    Process IDRiD images from data/raw/idrid/
    Handles both train and test subsets.
    Filters overlay annotation images — keeps only main fundus photos.
    """
    failed = []
    count  = 0

    for subset, csv_name in [('train', 'train_labels.csv'), ('test', 'test_labels.csv')]:
        raw_img_dir = RAW_DIR / 'idrid' / f'{subset}_images'
        csv_path    = RAW_DIR / 'idrid' / csv_name

        if not raw_img_dir.exists() or not csv_path.exists():
            print(f'  ⚠️  IDRiD {subset} not found, skipping')
            continue

        df = pd.read_csv(csv_path)
        df = df.rename(columns={'Image name': 'id_code'})

        # Keep only main fundus images (IDRiD_001, not IDRiD_001_L14)
        valid_ids = {
            f.stem for f in raw_img_dir.glob('*.jpg')
            if len(f.stem.split('_')) == 2
        }
        df = df[df['id_code'].isin(valid_ids)].reset_index(drop=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'  IDRiD {subset}'):
            src  = raw_img_dir  / f"{row['id_code']}.jpg"
            dest = proc_img_dir / f"idrid_{row['id_code']}.png"  # prefix idrid_

            if dest.exists():
                count += 1
                continue

            result = preprocess_image(src, size=size)
            if result is None:
                failed.append(row['id_code'])
                continue

            cv2.imwrite(str(dest), result)
            count += 1

    return count, failed


# ─────────────────────────────────────────────────────────────────────────────
# Build unified CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_csv(proc_dir: Path) -> pd.DataFrame:
    """
    Creates a single unified train.csv in data/processed/ covering
    both APTOS and IDRiD with columns:
      id_code       : prefixed filename stem (aptos_xxx or idrid_xxx)
      diagnosis     : grade 0–4
      source        : 'aptos' or 'idrid'
      macular_edema : 0–2 (IDRiD only, NaN for APTOS)
      caption       : clinical text (IDRiD only, NaN for APTOS)
    """
    rows = []

    # ── APTOS ─────────────────────────────────────────────────────────────────
    aptos_df = pd.read_csv(RAW_DIR / 'aptos' / 'train.csv')
    for _, row in aptos_df.iterrows():
        rows.append({
            'id_code'      : f"aptos_{row['id_code']}",
            'diagnosis'    : int(row['diagnosis']),
            'source'       : 'aptos',
            'macular_edema': None,
            'caption'      : None,
        })

    # ── IDRiD ─────────────────────────────────────────────────────────────────
    for subset, csv_name in [('train', 'train_labels.csv'), ('test', 'test_labels.csv')]:
        csv_path    = RAW_DIR / 'idrid' / csv_name
        raw_img_dir = RAW_DIR / 'idrid' / f'{subset}_images'

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path).rename(columns={
            'Image name'           : 'id_code',
            'Retinopathy grade'    : 'diagnosis',
            'Risk of macular edema': 'macular_edema',
            'Captions'             : 'caption',
        })

        valid_ids = {
            f.stem for f in raw_img_dir.glob('*.jpg')
            if len(f.stem.split('_')) == 2
        }
        df = df[df['id_code'].isin(valid_ids)]

        for _, row in df.iterrows():
            rows.append({
                'id_code'      : f"idrid_{row['id_code']}",
                'diagnosis'    : int(row['diagnosis']),
                'source'       : 'idrid',
                'macular_edema': row.get('macular_edema', None),
                'caption'      : row.get('caption', None),
            })

    unified  = pd.DataFrame(rows)
    out_path = proc_dir / 'train.csv'
    unified.to_csv(out_path, index=False)

    print(f'\nUnified CSV saved → {out_path}')
    print(f'  Total : {len(unified)}')
    print(f'  APTOS : {len(unified[unified.source == "aptos"])}')
    print(f'  IDRiD : {len(unified[unified.source == "idrid"])}')
    return unified


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process_all(size: int = IMG_SIZE) -> None:

    proc_img_dir = PROC_DIR / 'train_images'
    proc_img_dir.mkdir(parents=True, exist_ok=True)

    print(f'Output folder : {proc_img_dir}')
    print(f'Target size   : {size}×{size}px')
    print(f'Pipeline      : Crop → Resize → CLAHE → Ben Graham')
    print()

    print('Processing APTOS...')
    aptos_count, aptos_failed = process_aptos(proc_img_dir, size)

    print('\nProcessing IDRiD...')
    idrid_count, idrid_failed = process_idrid(proc_img_dir, size)

    print('\nBuilding unified CSV...')
    unified = build_unified_csv(PROC_DIR)

    total_failed = len(aptos_failed) + len(idrid_failed)
    grade_labels = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}

    print()
    print('=' * 50)
    print('  PREPROCESSING COMPLETE')
    print('=' * 50)
    print(f'  APTOS processed  : {aptos_count}')
    print(f'  IDRiD processed  : {idrid_count}')
    print(f'  Total processed  : {aptos_count + idrid_count}')
    print(f'  Failed           : {total_failed}')
    print()
    print('  Grade breakdown:')
    for g in range(5):
        c = len(unified[unified['diagnosis'] == g])
        print(f'    Grade {g} ({grade_labels[g]:<18}): {c}')
    print('=' * 50)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocess APTOS + IDRiD images')
    p.add_argument('--size', type=int, default=IMG_SIZE,
                   help=f'Output image size in pixels (default: {IMG_SIZE})')
    args = p.parse_args()
    process_all(size=args.size)
