"""
config.py
─────────
Global constants and hyperparameters for the InceptionV3 DR classifier.
All other modules import from here — change once, applies everywhere.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Kaggle input path — update if running locally
BASE_DIR = Path('/kaggle/input/datasets/arnavgandhi10000/aptos-idrid-combined-processed-v1')
IMG_DIR        = BASE_DIR / 'train_images'
CSV_PATH       = BASE_DIR / 'train.csv'

# Output paths (Kaggle working directory)
OUTPUT_DIR     = Path('/kaggle/working')
MODEL_DIR      = OUTPUT_DIR / 'models'
RESULTS_DIR    = OUTPUT_DIR / 'results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE       = 299          # InceptionV3 native input size
IMG_CHANNELS   = 3            # RGB
INPUT_SHAPE    = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_CLASSES    = 5            # Grades 0–4
CLASS_NAMES    = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
RANDOM_SEED    = 42

# Train / Val / Test split fractions
VAL_SPLIT      = 0.2          # paper uses 0.2 val from training data
TEST_SPLIT     = 0.15         # held-out test set

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32

# Phase 1 — warmup (frozen backbone)
WARMUP_EPOCHS  = 5
WARMUP_LR      = 1e-3

# Phase 2 — fine-tuning (unfrozen backbone)
FINETUNE_EPOCHS = 30
FINETUNE_LR     = 1e-5        # very low — prevents destroying pretrained weights

# ── Callbacks ─────────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE   = 7     # stop if val_loss doesn't improve
REDUCE_LR_PATIENCE    = 3     # reduce LR after 3 stagnant epochs
REDUCE_LR_FACTOR      = 0.3   # multiply LR by 0.3 on plateau
MIN_LR                = 1e-7

# ── Augmentation ──────────────────────────────────────────────────────────────
ROTATION_RANGE        = 360   # full rotation — retinas have no orientation
HORIZONTAL_FLIP       = True
VERTICAL_FLIP         = True

# ── Oversampling ──────────────────────────────────────────────────────────────
# All minority classes are upsampled to match the majority class (No DR = 1805)
OVERSAMPLE_TARGET     = None  # None = auto (uses majority class count)
