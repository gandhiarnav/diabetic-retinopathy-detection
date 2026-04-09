"""
config.py
─────────
All paths and hyperparameters in one place.
Every other module imports from here.
"""

from pathlib import Path

# ── Kaggle dataset paths ──────────────────────────────────────────────────────
# Your IDRiD dataset on Kaggle — update dataset name if different
DATA_DIR      = Path('/kaggle/input/datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid2')
IMAGE_DIR     = DATA_DIR / 'IDRiD' / 'train' / 'images'
TRAIN_CSV_PATH = DATA_DIR / 'IDRiD' / 'train' / 'annotations.csv'
TEST_CSV_PATH  = DATA_DIR / 'IDRiD' / 'test' / 'annotations.csv'
CSV_PATH      = TRAIN_CSV_PATH

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path('/kaggle/working')
MODEL_DIR   = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # ConvNext works well at 224

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_DR_CLASSES    = 5      # Grades 0–4
NUM_EDEMA_CLASSES = 3      # Risk 0–2
RANDOM_SEED       = 42
VAL_SPLIT         = 0.2    # 80/20 train/val split

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64   # was 32 — larger batches = GPU works longer per load
EPOCHS      = 10            # notebook reaches 0.95 kappa in just 5 epochs
LR          = 1e-4         # Adam LR for ConvNext

# Differential LRs for MultimodalModel (EfficientNet backbone)
BACKBONE_LR = 1e-5
HEAD_LR     = 1e-4

# ── Loss weights ──────────────────────────────────────────────────────────────
# Combined loss: 70% DR grading + 30% macular edema
DR_LOSS_WEIGHT    = 0.7
EDEMA_LOSS_WEIGHT = 0.3
