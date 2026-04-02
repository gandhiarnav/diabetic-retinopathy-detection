import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer

# Import your custom modules
from multimodal_model import MultimodalRetinopathyModel
from clinical_rules import generate_clinical_summary

def predict_single_patient(image_path, text_caption="", model_path='models/best_multimodal_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the Model
    model = MultimodalRetinopathyModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Process the Image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # 3. Process the Text
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    encoding = tokenizer(
        text_caption, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 4. Make the Prediction
    with torch.no_grad():
        outputs = model(image_tensor, input_ids, attention_mask)
        _, predicted_class = torch.max(outputs, 1)
        predicted_grade = predicted_class.item()

    # 5. Fetch the Clinical Summary
    summary = generate_clinical_summary(predicted_grade)

    # 6. Print the beautiful output
    print("\n" + "="*50)
    print(" 🏥 PATIENT DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Predicted Grade : {predicted_grade}")
    print(f"Diagnosis       : {summary['Diagnosis']}")
    print(f"Follow-up Timing: {summary['Timeline']}")
    print(f"Action Required : {summary['Action']}")
    print("="*50 + "\n")

if __name__ == '__main__':
    # Test it out! Just point it to any image in your raw dataset folder
    sample_image = 'data/raw/idrid/test_images/IDRiD_001.jpg' 
    sample_caption = "Severe diabetic retinopathy with extensive retinal lesions..."
    
    predict_single_patient(sample_image, sample_caption)
