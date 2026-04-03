"""
step3_model.py
===============
Multimodal DR model with a custom Cross-Modal Attention Fusion layer.

Architecture:
  Image branch  : EfficientNet-B4  → 1792-dim features
  Text branch   : ClinicalBERT     →  768-dim features
  Fusion        : CrossModalAttentionFusion (custom layer)
                  → learns which image regions align with clinical text
  Head          : MLP → 5 DR classes

The CrossModalAttentionFusion layer is the key novelty:
  Instead of naive concatenation, it uses scaled dot-product attention
  so each image feature attends to text features (and vice versa),
  producing a context-aware joint representation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel


# ================================================================
# CUSTOM LAYER: Cross-Modal Attention Fusion
# ================================================================
class CrossModalAttentionFusion(nn.Module):
    """
    Custom neural layer that fuses image and text features using
    scaled dot-product cross-attention.

    Given:
        img_feat  : (B, d_img)   — e.g., 1792-dim from EfficientNet-B4
        text_feat : (B, d_text)  — e.g.,  768-dim from ClinicalBERT

    Step 1 — Project both modalities into a shared d_model space.
    Step 2 — Image features act as Query; text features as Key & Value.
             This asks: "Which parts of the clinical text description
             are relevant to what the image encoder extracted?"
    Step 3 — Symmetrically: text as Query, image as Key & Value.
             This asks: "Which visual features align with the clinical
             language used in the caption?"
    Step 4 — Concatenate both attended outputs → feed to LayerNorm.

    Output : (B, 2 * d_model)
    """

    def __init__(self, d_img=1792, d_text=768, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Project each modality into the shared space
        self.img_proj  = nn.Linear(d_img,  d_model)
        self.text_proj = nn.Linear(d_text, d_model)

        # Cross-attention: image queries text
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: text queries image
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feat, text_feat):
        # Project to shared d_model space — add sequence dim of 1
        # because MultiheadAttention expects (B, seq_len, d_model)
        img_proj  = self.img_proj(img_feat).unsqueeze(1)    # (B, 1, d_model)
        text_proj = self.text_proj(text_feat).unsqueeze(1)  # (B, 1, d_model)

        # Image attends to text: "what does the text say about what I see?"
        img_attended, _ = self.img_to_text_attn(
            query=img_proj, key=text_proj, value=text_proj
        )
        img_attended = self.norm1(img_proj + self.dropout(img_attended)).squeeze(1)

        # Text attends to image: "what visual evidence supports my description?"
        text_attended, _ = self.text_to_img_attn(
            query=text_proj, key=img_proj, value=img_proj
        )
        text_attended = self.norm2(text_proj + self.dropout(text_attended)).squeeze(1)

        # Concatenate both attended representations
        fused = torch.cat([img_attended, text_attended], dim=1)  # (B, 2*d_model)
        return fused


# ================================================================
# FULL MODEL
# ================================================================
class MultimodalRetinopathyModel(nn.Module):
    def __init__(
        self,
        text_model_name='emilyalsentzer/Bio_ClinicalBERT',
        num_classes=5,
        dropout_rate=0.3,
        d_model=512,
        num_heads=8,
    ):
        super().__init__()

        # ── 1. Image Branch: EfficientNet-B4 ────────────────────────
        self.image_encoder = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT
        )
        image_feature_dim = self.image_encoder.classifier[1].in_features  # 1792
        self.image_encoder.classifier = nn.Identity()

        # ── 2. Text Branch: ClinicalBERT ────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_feature_dim  = self.text_encoder.config.hidden_size  # 768

        # ── 3. Custom Cross-Modal Attention Fusion Layer ─────────────
        self.fusion = CrossModalAttentionFusion(
            d_img=image_feature_dim,
            d_text=text_feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
        )

        # Output of fusion = 2 * d_model
        fused_dim = 2 * d_model

        # ── 4. Classification Head ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),                        # GELU smoother than ReLU for attention models
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        # Image features: (B, 1792)
        img_features = self.image_encoder(image)

        # Text features: (B, 768) — [CLS] pooled output
        text_out      = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.pooler_output

        # Cross-modal attention fusion: (B, 1024)
        fused = self.fusion(img_features, text_features)

        # Classification logits: (B, 5)
        logits = self.classifier(fused)
        return logits
