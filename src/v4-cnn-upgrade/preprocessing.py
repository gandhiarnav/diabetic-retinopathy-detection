"""
preprocessing.py
────────────────
Image cleaning functions following the paper's preprocessing pipeline.

Pipeline per image:
  1. Load image (BGR → RGB)
  2. Auto-crop black borders
  3. Circle crop (isolate retinal disc)
  4. Gaussian blur (reduce sensor noise)
  5. Resize to 299×299

Functions are modular — each step can be used independently.
"""

import cv2
import numpy as np
from pathlib import Path
from config import IMG_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Individual preprocessing steps
# ─────────────────────────────────────────────────────────────────────────────

def auto_crop(img: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Remove uninformative black borders around the retinal image.

    Fundus photographs often have large black regions where the camera
    didn't capture any tissue. These confuse the model and waste pixels.

    Args:
        img       : RGB image as numpy array
        tolerance : pixel brightness below this is treated as black

    Returns:
        Cropped image with black borders removed
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img  # all-black guard

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return img[r_min:r_max + 1, c_min:c_max + 1]


def circle_crop(img: np.ndarray) -> np.ndarray:
    """
    Isolate the circular retinal disc by masking everything outside it.

    Fundus cameras produce circular images — the corners are always black
    and uninformative. Circle cropping ensures the model focuses entirely
    on retinal tissue and preserves more of the zoomed-in informative region.

    Steps:
      1. Find the largest inscribed circle in the image
      2. Create a circular mask
      3. Apply mask — corners become black
      4. Crop to bounding box of the circle

    Returns:
        Square-cropped image with circular retina mask applied
    """
    h, w  = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(cx, cy)

    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)

    # Apply mask
    masked = img.copy()
    masked[mask == 0] = 0

    # Crop to circle bounding box
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius)

    return masked[y1:y2, x1:x2]


def apply_gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce camera sensor noise.

    Different fundus cameras introduce different types of noise based on
    their sensors. Gaussian blur smooths out sharp noise artefacts while
    preserving the edges of blood vessels and lesions that the model
    needs to detect.

    Args:
        img   : RGB image
        sigma : standard deviation of Gaussian kernel (higher = more blur)

    Returns:
        Blurred image
    """
    # Kernel size must be odd — scale with sigma
    k = max(3, int(sigma * 6) | 1)   # | 1 ensures odd number
    return cv2.GaussianBlur(img, (k, k), sigma)


def preprocess_image(
    img_path: Path | str,
    size    : int = IMG_SIZE,
) -> np.ndarray | None:
    """
    Full preprocessing pipeline for a single image.
    Applies all steps in the correct order as described in the paper.

    Steps:
      1. Load with OpenCV (BGR)
      2. Convert BGR → RGB  (InceptionV3 was trained on RGB)
      3. Auto-crop black borders
      4. Circle crop
      5. Gaussian blur
      6. Resize to target size

    Args:
        img_path : path to image file (.png or .jpg)
        size     : target output size (default 299 for InceptionV3)

    Returns:
        Preprocessed RGB numpy array (size, size, 3) or None if load fails
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # ── BGR → RGB ─────────────────────────────────────────────────────────────
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Auto-crop black borders ───────────────────────────────────────────────
    img = auto_crop(img)

    # ── Circle crop ───────────────────────────────────────────────────────────
    img = circle_crop(img)

    # ── Gaussian blur ─────────────────────────────────────────────────────────
    img = apply_gaussian_blur(img, sigma=1.0)

    # ── Resize ────────────────────────────────────────────────────────────────
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    return img.astype(np.uint8)
