import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from tqdm import tqdm

# Import your custom modules
from multimodal_dataset_loader import IDRiDMultimodalDataset
from multimodal_model import MultimodalRetinopathyModel

def evaluate_model():
    # ==========================================
    # 1. Configuration & Paths
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    BATCH_SIZE = 16
    
    # Kaggle Paths
    VAL_CSV = 'data/processed/idrid/test_labels.csv'
    VAL_IMG_DIR = 'data/processed/idrid/test_images/'
    MODEL_PATH = 'models/best_multimodal_model.pth'
    RESULTS_DIR = 'results/results_multimodal/'

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading (Validation Set Only)
    # ==========================================
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])

    print("Loading test dataset...")
    val_dataset = IDRiDMultimodalDataset(csv_file=VAL_CSV, image_dir=VAL_IMG_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ==========================================
    # 3. Load the Saved Model
    # ==========================================
    print("Loading saved model weights...")
    model = MultimodalRetinopathyModel().to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode (turns off dropout, batchnorm updates)

    # ==========================================
    # 4. Inference Loop
    # ==========================================
    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==========================================
    # 5. Generate Metrics and Reports
    # ==========================================
    print("\nGenerating Evaluation Reports...")
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Generate Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])
    
    # Save Text Report
    report_path = os.path.join(RESULTS_DIR, 'multimodal_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Diabetic Retinopathy Multimodal Model Evaluation\n")
        f.write("=================================================\n\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Quadratic Weighted Kappa: {kappa:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 30 + "\n")
        f.write(class_report)
    print(f"✅ Saved text report to {report_path}")

    # Generate Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2', '3', '4'], 
                yticklabels=['0', '1', '2', '3', '4'])
    plt.title(f'Confusion Matrix (Kappa: {kappa:.4f}, Acc: {accuracy*100:.1f}%)')
    plt.ylabel('Actual Retinopathy Grade')
    plt.xlabel('Predicted Retinopathy Grade')
    
    # Save Heatmap Image
    heatmap_path = os.path.join(RESULTS_DIR, 'multimodal_confusion_matrix.png')
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved heatmap to {heatmap_path}")

    print("\nEvaluation Complete! Check the 'results' folder.")

if __name__ == '__main__':
    evaluate_model()
