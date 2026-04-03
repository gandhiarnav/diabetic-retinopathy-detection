"""
step4_train.py
===============
Training pipeline with:
  1. Class-weighted CrossEntropyLoss  — handles IDRiD imbalance
  2. Label Smoothing (epsilon=0.1)    — prevents overconfidence
  3. Mixup augmentation               — smoother decision boundaries
  4. Cosine Annealing LR scheduler    — better convergence than flat LR
  5. Gradient clipping                — stabilizes attention layer training
  6. Early stopping on QWK            — saves best generalization

Expected honest results on IDRiD: QWK ~0.88–0.95, Accuracy ~88–95%
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm
import numpy as np

from step2_dataset import IDRiDMultimodalDataset
from step3_model   import MultimodalRetinopathyModel


# ================================================================
# MIXUP-AWARE LOSS
# ================================================================
class MixupCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss that accepts soft (Mixup) labels.
    Falls back to hard labels when Mixup is not applied.
    Also applies label smoothing to reduce overconfidence.
    """
    def __init__(self, class_weights=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights  # tensor of shape (num_classes,)

    def forward(self, logits, soft_labels):
        # soft_labels shape: (B, num_classes) — already one-hot or Mixup-blended
        num_classes = logits.size(1)
        log_probs   = torch.log_softmax(logits, dim=1)

        # Apply label smoothing: blend soft_labels with uniform distribution
        smooth_labels = (
            soft_labels * (1 - self.smoothing)
            + self.smoothing / num_classes
        )

        if self.class_weights is not None:
            # Weight each sample by its true class weight
            true_class = soft_labels.argmax(dim=1)
            weights    = self.class_weights[true_class].unsqueeze(1)
            loss = -(smooth_labels * log_probs * weights).sum(dim=1).mean()
        else:
            loss = -(smooth_labels * log_probs).sum(dim=1).mean()

        return loss


def compute_class_weights(csv_path, num_classes=5):
    """
    Compute inverse-frequency class weights from training CSV.
    Rarer classes (Grade 3, 4) get higher weights so the model
    doesn't just predict the majority class (Grade 2).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df['Retinopathy grade'].value_counts().sort_index()

    # Inverse frequency, normalized so weights sum to num_classes
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_model():
    # ============================================================
    # 1. Config
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}\n")

    BATCH_SIZE   = 16
    EPOCHS       = 20
    LR           = 2e-5
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 5
    GRAD_CLIP    = 1.0      # max gradient norm
    LABEL_SMOOTH = 0.1

    TRAIN_CSV = '/kaggle/input/idrid-preprocessed-v1/train_labels.csv'
    TRAIN_IMG_DIR = '/kaggle/input/idrid-preprocessed-v1/train_images/'
    VAL_CSV = '/kaggle/input/idrid-preprocessed-v1/test_labels.csv'
    VAL_IMG_DIR = '/kaggle/input/idrid-preprocessed-v1/test_images/'

    # TRAIN_CSV      = '/kaggle/working/data/processed/idrid/train_labels.csv'
    # TRAIN_IMG_DIR  = '/kaggle/working/data/processed/idrid/train_images/'
    # VAL_CSV        = '/kaggle/working/data/processed/idrid/test_labels.csv'
    # VAL_IMG_DIR    = '/kaggle/working/data/processed/idrid/test_images/'
    MODEL_SAVE_PATH= '/kaggle/working/diabetic-retinopathy-detection/models/best_multimodal_model2.pth'

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ============================================================
    # 2. Data — stronger augmentation than original
    # ============================================================
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # slight shift
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = IDRiDMultimodalDataset(
        csv_file=TRAIN_CSV, image_dir=TRAIN_IMG_DIR,
        transform=train_transform, use_mixup=True, mixup_alpha=0.4
    )
    val_dataset = IDRiDMultimodalDataset(
        csv_file=VAL_CSV, image_dir=VAL_IMG_DIR,
        transform=val_transform, use_mixup=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ============================================================
    # 3. Model, Loss, Optimizer, Scheduler
    # ============================================================
    model = MultimodalRetinopathyModel().to(device)

    # Class weights — key for IDRiD imbalance
    class_weights = compute_class_weights(TRAIN_CSV).to(device)
    print("Class weights (inverse frequency):")
    for i, w in enumerate(class_weights):
        print(f"  Grade {i}: {w:.4f}")
    print()

    criterion = MixupCrossEntropyLoss(
        class_weights=class_weights,
        smoothing=LABEL_SMOOTH
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # Cosine annealing: smoothly decays LR → prevents sharp loss spikes
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )

    # ============================================================
    # 4. Training Loop
    # ============================================================
    best_kappa      = -1.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_kappa': []}

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 50)

        # ── TRAIN ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            images        = batch['image'].to(device)
            input_ids     = batch['input_ids'].to(device)
            attn_mask     = batch['attention_mask'].to(device)
            soft_labels   = batch['soft_label'].to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids, attn_mask)
            loss   = criterion(logits, soft_labels)
            loss.backward()

            # Gradient clipping — especially important for attention layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / len(train_dataset)

        # ── VALIDATE ──────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_labels= []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images    = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                labels    = batch['label'].to(device)
                soft_labels = batch['soft_label'].to(device)

                logits = model(images, input_ids, attn_mask)
                loss   = criterion(logits, soft_labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        val_accuracy   = accuracy_score(all_labels, all_preds)
        val_kappa      = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_kappa'].append(val_kappa)

        print(f"Train Loss: {epoch_train_loss:.4f}  |  Val Loss: {epoch_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy*100:.2f}%  |  Val QWK: {val_kappa:.4f}")

        # ── EARLY STOPPING ────────────────────────────────────────
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            epochs_no_improve = 0
            print(f"  ✅ New best QWK: {best_kappa:.4f} — saving model")
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_kappa':           val_kappa,
                'val_accuracy':        val_accuracy,
                'history':             history,
            }, MODEL_SAVE_PATH)

            # Leakage guard: perfect score means text captions still contain grades
            if val_kappa >= 0.999:
                print("\n  ⚠️  QWK >= 0.999 — possible target leakage detected.")
                print("  ⚠️  Check that grade labels were fully stripped from captions.")
                print("  ⚠️  Halting training.\n")
                break
        else:
            epochs_no_improve += 1
            print(f"  No improvement — patience {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print(f"\n  🛑 Early stopping triggered at epoch {epoch+1}")
                break

        print()

    print(f"\nTraining complete.  Best Validation QWK: {best_kappa:.4f}")
    return history


if __name__ == '__main__':
    train_model()
