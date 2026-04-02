import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm

# Import your custom modules
from multimodal_dataset_loader import IDRiDMultimodalDataset
from multimodal_model import MultimodalRetinopathyModel

def train_model():
    # ==========================================
    # 1. Configuration & Hyperparameters
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    BATCH_SIZE = 16 
    EPOCHS = 15
    LEARNING_RATE = 2e-5 
    
    # File paths - Updated for Kaggle Preprocessed Data
    TRAIN_CSV = '/kaggle/working/data/processed/idrid/train_labels.csv'
    TRAIN_IMG_DIR = '/kaggle/working/data/processed/idrid/train_images/'
    VAL_CSV = '/kaggle/working/data/processed/idrid/test_labels.csv'
    VAL_IMG_DIR = '/kaggle/working/data/processed/idrid/test_images/'
    MODEL_SAVE_PATH = '/kaggle/working/diabetic-retinopathy-detection/models/best_multimodal_model.pth'

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ==========================================
    # 2. Data Loading & Augmentation
    # ==========================================
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = IDRiDMultimodalDataset(csv_file=TRAIN_CSV, image_dir=TRAIN_IMG_DIR, transform=train_transform)
    val_dataset = IDRiDMultimodalDataset(csv_file=VAL_CSV, image_dir=VAL_IMG_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ==========================================
    # 3. Model, Loss, and Optimizer
    # ==========================================
    model = MultimodalRetinopathyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # ==========================================
    # 4. Training Loop with Early Stopping
    # ==========================================
    best_kappa = -1.0
    patience = 3 # Stop if no improvement after 3 epochs
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 20)
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        
        # Calculate Metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy*100:.2f}% | Val QWK Score: {val_kappa:.4f}")

        # --- EARLY STOPPING & SAVING LOGIC ---
        if val_kappa > best_kappa:
            print(f"*** New Best QWK Score! Saving model to {MODEL_SAVE_PATH} ***")
            best_kappa = val_kappa
            epochs_no_improve = 0 # Reset patience counter
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
            }, MODEL_SAVE_PATH)
            
            # Target Leakage Kill-Switch
            if val_kappa == 1.0:
                print("⚠️ Perfect 1.0 score reached. The model is reading the text targets. Triggering immediate early stopping.")
                break 
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"\n🛑 Early Stopping Triggered! Halting training to prevent overfitting.")
                break

    print("\nTraining Complete.")
    print(f"Best Validation QWK Score: {best_kappa:.4f}")
    print("Model saved and ready for local evaluation.")

if __name__ == '__main__':
    train_model()
