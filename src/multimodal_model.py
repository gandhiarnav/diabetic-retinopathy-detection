import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

class MultimodalRetinopathyModel(nn.Module):
    def __init__(self, text_model_name='emilyalsentzer/Bio_ClinicalBERT', num_classes=5, dropout_rate=0.3):
        super(MultimodalRetinopathyModel, self).__init__()
        
        # ==========================================
        # 1. Image Branch: EfficientNet-B4
        # ==========================================
        # Load pre-trained EfficientNet
        self.image_encoder = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        
        # Extract the number of features before the final classification layer
        # For EfficientNet-B4, this is 1792
        image_feature_dim = self.image_encoder.classifier[1].in_features
        
        # Remove the final classification layer to just output the raw feature vector
        self.image_encoder.classifier = nn.Identity()

        # ==========================================
        # 2. Text Branch: ClinicalBERT
        # ==========================================
        # Load pre-trained BERT model (only the base, no classification head)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Standard BERT outputs a 768-dimensional feature vector
        text_feature_dim = self.text_encoder.config.hidden_size 

        # ==========================================
        # 3. Multimodal Fusion Head
        # ==========================================
        # Combine the feature dimensions
        total_feature_dim = image_feature_dim + text_feature_dim
        
        # Build a custom multi-layer perceptron (MLP) for the final prediction
        self.fusion_head = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes) # Outputs logits for the 5 classes (0, 1, 2, 3, 4)
        )

    def forward(self, image, input_ids, attention_mask):
        # 1. Extract Image Features
        # Shape: (batch_size, 1792)
        img_features = self.image_encoder(image)
        
        # 2. Extract Text Features
        # We use the 'pooler_output' which is the representation of the [CLS] token
        # Shape: (batch_size, 768)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # 3. Fuse Features
        # Concatenate along the feature dimension (dim=1)
        # Shape: (batch_size, 1792 + 768)
        combined_features = torch.cat((img_features, text_features), dim=1)
        
        # 4. Final Classification
        # Shape: (batch_size, 5)
        logits = self.fusion_head(combined_features)
        
        return logits