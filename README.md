# Diabetic Retinopathy Detection using Deep Learning

## Overview
This project builds a deep learning model to detect diabetic retinopathy from retinal fundus images.

## Dataset
Dataset from Kaggle:
APTOS 2019 Blindness Detection 
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

IDRiD: Excellent for "Early Detection" as it provides pixel-level annotations for lesions.
https://www.kaggle.com/datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid2

Architecture Explanation

3.1 Dataset
The study uses the Indian Diabetic Retinopathy Image Dataset (IDRiD), which provides 413 fundus photographs annotated with five-level DR grading (0–4) and a three-level macular edema risk score (0–2). Each image is also accompanied by a structured natural-language clinical caption describing the pathological findings.

3.2 Lesion-Aware Data Expansion
A key contribution of this work is the use of IDRiD's lesion overlay images as additional training samples. For each original fundus photograph, the dataset provides multiple lesion annotation overlays — spatially registered masks highlighting specific pathological structures. These overlay images are treated as distinct training instances carrying the same grade label as their base image. This expands the effective training set from 413 images to 3,304 samples without any synthetic generation, as all images are real pathological renderings derived from the original fundus photographs.

3.3 Data Preparation
The dataset is split into 80% training and 20% validation using stratified sampling to preserve the class distribution. The lesion-aware expansion is applied exclusively to the training set; the validation set retains original images only. Each image is resized to 224×224 pixels and normalised using ImageNet channel statistics (mean and standard deviation per channel).

3.4 Model Architectures
Two independent model architectures were trained and compared.
Model A — EfficientNetB0 with multimodal fusion. The EfficientNetB0 backbone (pretrained on ImageNet) extracts a 1,280-dimensional feature vector from the input image. A parallel text branch processes optional CLIP-encoded caption embeddings (512-dimensional) through a linear layer, ReLU activation, and batch normalisation to produce a 128-dimensional text representation. The image and text features are concatenated (1,408 dimensions total) and passed through a fusion layer consisting of a linear projection to 256 dimensions, ReLU activation, and 30% dropout. When captions are unavailable, the text branch receives a zero vector, making the model operate in image-only mode.
Model B — ConvNeXt-Tiny. The ConvNeXt-Tiny backbone (pretrained on ImageNet) produces a feature map that is flattened to a 768-dimensional vector. No text branch is used. This architecture is more modern than EfficientNet, employing large convolutional kernels, LayerNorm, and GELU activations, which make it well-suited to capturing the fine texture differences that distinguish DR severity grades.

3.5 Dual-Head Output
Both models share the same output structure: two independent linear classification heads applied to the shared feature representation. The DR head projects to five classes corresponding to grades 0–4 (No DR, Mild, Moderate, Severe, and Proliferative DR). The edema head projects to three classes corresponding to macular edema risk levels (none, moderate, clinically significant). This multi-task design allows the model to learn complementary pathological signals simultaneously.

3.6 Training Objective
The combined training loss is a weighted sum of two cross-entropy terms:
L = 0.7 × L_DR + 0.3 × L_edema
Class-frequency inverse weighting is applied to both loss terms to mitigate the effect of class imbalance. Models are trained using the Adam optimiser with mixed-precision (AMP) for computational efficiency and gradient clipping at norm 1.0. Training runs for up to 20 epochs with early stopping triggered after 3 consecutive epochs without validation kappa improvement, preserving the best checkpoint.

3.7 Evaluation
Primary evaluation uses Quadratic Weighted Kappa (QWK), the standard metric for ordinal DR grading tasks. QWK penalises large grade discrepancies more heavily than small ones and is insensitive to class imbalance. Additional metrics reported include accuracy, per-class precision, recall, and F1 score, as well as ROC-AUC per grade using a one-vs-rest strategy.

3.8 Clinical Triage Output
At inference time, the predicted DR grade and edema risk are passed to a rule-based clinical triage matrix. Edema grade acts as an escalation factor — higher edema risk elevates the urgency tier regardless of DR grade. The system outputs one of four urgency levels (routine, soon, urgent, emergency), an actionable recommendation, and a clinical timeframe for specialist consultation. This bridges the gap between model output and clinical decision support.

## Results

![Confusion Matrix](results/v5-indian-results/02_confusion_matrix_pct)

## How to Run

git clone
pip install -r requirements.txt
python src/train.py
