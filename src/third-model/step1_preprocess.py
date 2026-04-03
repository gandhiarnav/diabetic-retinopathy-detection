"""
step1_preprocess.py
====================
Preprocessing pipeline for IDRiD dataset.
- Resizes images to 256x256 with CLAHE enhancement (green channel focus)
- Strips DR grade labels from captions to prevent target leakage
- Saves cleaned CSVs and processed images

Run this FIRST before any training.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================
# GRADE LABEL PATTERNS TO STRIP FROM CAPTIONS
# These patterns cause target leakage in the text encoder
# ============================================================
GRADE_PATTERNS = [
    r'\bgrade\s*[0-4]\b',
    r'\bdr\s*grade\s*[0-4]\b',
    r'\bretinopathy\s*grade\s*[0-4]\b',
    r'\bno\s*(diabetic\s*)?retinopathy\b',
    r'\bmild\s*(non-?proliferative\s*)?(diabetic\s*)?retinopathy\b',
    r'\bmoderate\s*(non-?proliferative\s*)?(diabetic\s*)?retinopathy\b',
    r'\bsevere\s*(non-?proliferative\s*)?(diabetic\s*)?retinopathy\b',
    r'\bproliferative\s*(diabetic\s*)?retinopathy\b',
    r'\bnpdr\b',
    r'\bpdr\b',
    r'\bstage\s*[0-4]\b',
    r'\blevel\s*[0-4]\b',
    r'\bclass\s*[0-4]\b',
    r'\b(grade|severity)[:\-\s]+[0-4]\b',
]

def strip_grade_labels(text: str) -> str:
    """
    Remove explicit DR grade/severity labels from captions.
    This prevents ClinicalBERT from reading the answer directly.
    Retains all other clinical language (lesion descriptions, findings, etc.)
    """
    if pd.isna(text) or str(text).strip() == "":
        return "fundus image with retinal findings"

    text = str(text).lower()

    for pattern in GRADE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean up leftover punctuation / whitespace artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[\s,;:\-\.]+|[\s,;:\-\.]+$', '', text)

    # Fallback if caption is now empty
    if len(text) < 5:
        text = "fundus image with retinal findings"

    return text


def apply_clahe_green_channel(img_array: np.ndarray) -> np.ndarray:
    """
    Medical-grade preprocessing for fundus images:
    1. Extract green channel (highest contrast for DR lesions)
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
       to enhance microaneurysms and hemorrhages
    3. Merge back into RGB for compatibility with pre-trained models

    CLAHE is the standard preprocessing step in published DR detection
    literature (e.g., Gulshan et al., 2016; IDRiD Challenge baselines).
    """
    # Extract green channel — DR lesions (microaneurysms, hemorrhages)
    # are most visible in the green channel of fundus images
    green_channel = img_array[:, :, 1]

    # CLAHE parameters tuned for fundus images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)

    # Apply mild CLAHE to R and B channels too for balanced enhancement
    r_channel = clahe.apply(img_array[:, :, 0])
    b_channel = clahe.apply(img_array[:, :, 2])

    # Reconstruct RGB image with CLAHE-enhanced channels
    enhanced = np.stack([r_channel, enhanced_green, b_channel], axis=2)
    return enhanced


def process_dataset(raw_csv, raw_img_dir, proc_csv, proc_img_dir, img_size=(256, 256)):
    """
    Full preprocessing pipeline:
    - Strips grade labels from captions
    - Applies CLAHE green-channel enhancement
    - Resizes to target resolution
    """
    os.makedirs(proc_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(proc_csv), exist_ok=True)

    df = pd.read_csv(raw_csv)
    original_captions = df['Captions'].copy()

    print(f"\nStripping grade labels from captions in: {raw_csv}")
    df['Captions'] = df['Captions'].apply(strip_grade_labels)

    # Log a few examples so you can verify stripping worked
    print("\nCaption cleaning examples (first 3 rows):")
    for i in range(min(3, len(df))):
        print(f"  BEFORE: {str(original_captions.iloc[i])[:80]}")
        print(f"  AFTER : {str(df['Captions'].iloc[i])[:80]}")
        print()

    print(f"Applying CLAHE preprocessing and resizing to {img_size}...")
    failed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_name = str(row['Image name'])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'

        raw_path = os.path.join(raw_img_dir, img_name)
        proc_path = os.path.join(proc_img_dir, img_name)

        if os.path.exists(raw_path):
            try:
                # Load with PIL → convert to numpy for OpenCV processing
                with Image.open(raw_path) as img:
                    img = img.convert('RGB')
                    img_array = np.array(img)

                # Apply medical preprocessing (CLAHE + green channel)
                enhanced = apply_clahe_green_channel(img_array)

                # Resize using high-quality Lanczos for downscaling
                enhanced_pil = Image.fromarray(enhanced)
                enhanced_pil = enhanced_pil.resize(img_size, Image.Resampling.LANCZOS)
                enhanced_pil.save(proc_path, quality=95)

            except Exception as e:
                print(f"  Warning: Failed to process {img_name}: {e}")
                failed += 1
        else:
            print(f"  Warning: {raw_path} not found — skipping.")
            failed += 1

    df.to_csv(proc_csv, index=False)

    print(f"\nDone. Processed: {len(df) - failed}/{len(df)} images")
    print(f"Saved cleaned CSV to: {proc_csv}\n")

    # Print class distribution for awareness
    print("Class distribution:")
    print(df['Retinopathy grade'].value_counts().sort_index().to_string())
    print()


if __name__ == "__main__":
    # LOCAL PATHS:
    RAW_DIR = 'data/raw/idrid/'
    PROC_DIR = 'data/processed/idrid2/'
    
    # Train Data
    process_dataset(
        raw_csv=os.path.join(RAW_DIR, 'train_labels.csv'),
        raw_img_dir=os.path.join(RAW_DIR, 'train_images/'),
        proc_csv=os.path.join(PROC_DIR, 'train_labels.csv'),
        proc_img_dir=os.path.join(PROC_DIR, 'train_images/')
    )
    
    # Test Data
    process_dataset(
        raw_csv=os.path.join(RAW_DIR, 'test_labels.csv'),
        raw_img_dir=os.path.join(RAW_DIR, 'test_images/'),
        proc_csv=os.path.join(PROC_DIR, 'test_labels.csv'),
        proc_img_dir=os.path.join(PROC_DIR, 'test_images/')
    )

    print("All preprocessing complete. Data is ready for training.")
